# SPDX-License-Identifier: Apache-2.0
"""Joint audio-video denoising for MAGI-2's preview and refiner passes."""

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    run_cfg_parallel,
    run_two_branch_cfg_parallel,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    guidance as magi2_guidance,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class Magi2DenoisingStage(DenoisingStage):
    """Run one denoise loop over the packed video+audio sequence.

    ``forward`` is model-owned: the two modalities carry separate guidance scales
    and separate multistep scheduler state.
    """

    def __init__(
        self,
        *,
        transformer,
        pipeline=None,
        guidance_key: str = "",
        refiner_only: bool = False,
    ) -> None:
        # scheduler=None: the schedule is per-modality, built in the preparation stage.
        super().__init__(transformer=transformer, scheduler=None, pipeline=pipeline)
        self.guidance_key = guidance_key
        self.refiner_only = refiner_only

    def _owns_compile_warmup_lifecycle(self) -> bool:
        # The base tests whether forward is inherited, so not overriding this would
        # silently disable offload-during-compile rather than raise.
        return True

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def _scales(self, batch: Req) -> tuple[float, float, float | None]:
        params = batch.sampling_params
        if self.guidance_key == "refiner":
            return (
                params.refiner_guidance_scale,
                params.refiner_audio_guidance_scale,
                None,
            )
        return (
            params.guidance_scale,
            params.audio_guidance_scale,
            params.skimmed_guidance_scale if params.use_skimmed_guidance else None,
        )

    def _predict(
        self,
        *,
        video: torch.Tensor,
        audio: torch.Tensor | None,
        text: torch.Tensor,
        layout,
        coords: torch.Tensor,
        timestep: torch.Tensor,
        ref_patches: torch.Tensor | None = None,
        ref_special: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        extra = {}
        # Only the preview accepts ref images; the refiner's allowlist raises on them.
        if ref_patches is not None:
            extra = {"ref_patches": ref_patches, "ref_special": ref_special}
        return self.transformer(
            video_latents=video,
            audio_latents=audio,
            text_embeds=text,
            layout=layout,
            coords=coords,
            timestep=timestep,
            **extra,
        )

    def _build_cfg_policy(
        self,
        *,
        batch: Req,
        pipeline_config,
        prompt: torch.Tensor,
        negative: torch.Tensor,
        layout,
        uncond_layout,
        coords: torch.Tensor,
        uncond_coords: torch.Tensor,
        ref_patches: torch.Tensor | None,
        ref_special: torch.Tensor | None,
    ) -> CFGPolicy:
        """Build the two invariant MAGI-2 branch descriptions once per request."""
        image_kwargs = {}
        if ref_patches is not None:
            image_kwargs = {"ref_patches": ref_patches, "ref_special": ref_special}
        cfg_policy = getattr(pipeline_config, "cfg_policy", None) or CFGPolicy()
        return cfg_policy.build(
            batch,
            image_kwargs,
            {"text": prompt, "layout": layout, "coords": coords},
            {"text": negative, "layout": uncond_layout, "coords": uncond_coords},
        )

    def _predict_cfg_parallel(
        self,
        *,
        batch: Req,
        policy: CFGPolicy,
        pipeline_config,
        video: torch.Tensor,
        audio: torch.Tensor | None,
        step_t: torch.Tensor,
        video_scale: float,
        audio_scale: float,
        skimmed_scale: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run conditional/unconditional MAGI-2 branches on separate CFG ranks.

        The packed layout can have different text lengths per branch, so each
        branch builds its own per-token timestep.  Outputs remain full video/audio
        tensors after the model's SP gather, making the CFG all-reduce shape-safe.
        """
        if len(policy.branches) != 2:
            raise ValueError(
                "MAGI-2 CFG parallelism currently requires exactly two branches, "
                f"got {len(policy.branches)}"
            )

        def predict_branch(branch):
            branch.configure_batch(batch)
            branch_kwargs = branch.kwargs
            branch_t = packed_sequence.build_timesteps(
                layout=branch_kwargs["layout"], video_t=step_t, audio_t=step_t
            )
            predicted_video, predicted_audio = self._predict(
                video=video,
                audio=audio,
                text=branch_kwargs["text"],
                layout=branch_kwargs["layout"],
                coords=branch_kwargs["coords"],
                timestep=branch_t,
                ref_patches=branch_kwargs.get("ref_patches"),
                ref_special=branch_kwargs.get("ref_special"),
            )
            # The no-audio request returns ``None`` for the second model output;
            # keep the utility's tensor-only contract in that case.
            return (
                predicted_video if audio is None else (predicted_video, predicted_audio)
            )

        if skimmed_scale is None and get_classifier_free_guidance_world_size() == 2:
            cfg_scale = video_scale if audio is None else (video_scale, audio_scale)
            combined = run_two_branch_cfg_parallel(
                policy,
                predict_branch,
                cfg_scale,
                batch,
                pipeline_config,
            )
            batch.is_cfg_negative = False
            return (combined, None) if audio is None else combined

        # Skimmed guidance (and any future N-branch policy) needs both predictions
        # locally.  The generic dispatcher performs the required CFG all-gather.
        predictions = run_cfg_parallel(policy, predict_branch)
        positive = predictions[0]
        negative = predictions[1]
        if audio is None:
            combined_video = magi2_guidance.apply_guidance(
                latent=video,
                cond=positive,
                uncond=negative,
                guidance_scale=video_scale,
                skimmed_scale=skimmed_scale,
            )
            batch.is_cfg_negative = False
            return combined_video, None

        positive_video, positive_audio = positive
        negative_video, negative_audio = negative
        combined_video = magi2_guidance.apply_guidance(
            latent=video,
            cond=positive_video,
            uncond=negative_video,
            guidance_scale=video_scale,
            skimmed_scale=skimmed_scale,
        )
        combined_audio = None
        if positive_audio is not None and negative_audio is not None:
            combined_audio = magi2_guidance.apply_guidance(
                latent=audio,
                cond=positive_audio,
                uncond=negative_audio,
                guidance_scale=audio_scale,
                skimmed_scale=skimmed_scale,
            )
        batch.is_cfg_negative = False
        return combined_video, combined_audio

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self.refiner_only and not batch.extra["magi2_enable_refiner"]:
            return batch

        device = get_local_torch_device()
        video_scale, audio_scale, skimmed_scale = self._scales(batch)
        use_guidance = server_args.pipeline_config.should_use_guidance

        layout = batch.extra["magi2_layout"]
        coords = batch.extra["magi2_coords"]
        prompt = batch.prompt_embeds[0]
        negative = (
            batch.negative_prompt_embeds[0]
            if use_guidance and batch.negative_prompt_embeds
            else None
        )
        uncond_layout = batch.extra.get("magi2_layout_uncond", layout)
        uncond_coords = batch.extra.get("magi2_coords_uncond", coords)

        video = batch.latents
        audio = batch.audio_latents
        scheduler = batch.scheduler

        # Build invariant branch metadata once.  The normal CFG1 path below stays
        # byte-for-byte in its existing serial order when this flag is disabled.
        cfg_policy = None
        if (
            negative is not None
            and getattr(batch, "do_classifier_free_guidance", True)
            and getattr(server_args, "enable_cfg_parallel", False)
        ):
            cfg_policy = self._build_cfg_policy(
                batch=batch,
                pipeline_config=server_args.pipeline_config,
                prompt=prompt,
                negative=negative,
                layout=layout,
                uncond_layout=uncond_layout,
                coords=coords,
                uncond_coords=uncond_coords,
                ref_patches=(
                    batch.extra["magi2_ref_patches"]
                    if layout.ref_patch_index.numel()
                    else None
                ),
                ref_special=(
                    batch.extra["magi2_ref_special"]
                    if layout.ref_patch_index.numel()
                    else None
                ),
            )

        ref_patches = (
            batch.extra["magi2_ref_patches"] if layout.ref_patch_index.numel() else None
        )
        ref_special = (
            batch.extra["magi2_ref_special"] if ref_patches is not None else None
        )

        for step, timestep in enumerate(batch.timesteps):
            with (
                maybe_nvtx_range(f"denoising_step_{step}", self.current_use_nvtx),
                StageProfiler(
                    f"denoising_step_{step}",
                    logger=logger,
                    metrics=batch.metrics,
                    perf_dump_path_provided=batch.perf_dump_path is not None,
                    record_as_step=True,
                ),
            ):
                # Per token, not a scalar: text and ref-image rows must read zero.
                step_t = timestep.to(device)

                if cfg_policy is not None:
                    cond_video, cond_audio = self._predict_cfg_parallel(
                        batch=batch,
                        policy=cfg_policy,
                        pipeline_config=server_args.pipeline_config,
                        video=video,
                        audio=audio,
                        step_t=step_t,
                        video_scale=video_scale,
                        audio_scale=audio_scale,
                        skimmed_scale=skimmed_scale,
                    )
                else:
                    t = packed_sequence.build_timesteps(
                        layout=layout, video_t=step_t, audio_t=step_t
                    )
                    cond_video, cond_audio = self._predict(
                        video=video,
                        audio=audio,
                        text=prompt,
                        layout=layout,
                        coords=coords,
                        timestep=t,
                        ref_patches=ref_patches,
                        ref_special=ref_special,
                    )

                if negative is not None and cfg_policy is None:
                    uncond_video, uncond_audio = self._predict(
                        video=video,
                        audio=audio,
                        text=negative,
                        layout=uncond_layout,
                        coords=uncond_coords,
                        timestep=packed_sequence.build_timesteps(
                            layout=uncond_layout, video_t=step_t, audio_t=step_t
                        ),
                        ref_patches=ref_patches,
                        ref_special=ref_special,
                    )
                    cond_video = magi2_guidance.apply_guidance(
                        latent=video,
                        cond=cond_video,
                        uncond=uncond_video,
                        guidance_scale=video_scale,
                        skimmed_scale=skimmed_scale,
                    )
                    if cond_audio is not None and uncond_audio is not None:
                        cond_audio = magi2_guidance.apply_guidance(
                            latent=audio,
                            cond=cond_audio,
                            uncond=uncond_audio,
                            guidance_scale=audio_scale,
                            skimmed_scale=skimmed_scale,
                        )

                video = scheduler.step(cond_video, timestep, video, return_dict=False)[
                    0
                ]
                if cond_audio is not None:
                    audio = batch.extra["magi2_audio_scheduler"].step(
                        cond_audio, timestep, audio, return_dict=False
                    )[0]

            # Advance the torch profiler schedule just like the generic
            # denoising loop. The MAGI-2 loop is custom, so this hook is not
            # inherited automatically.
            if not batch.is_warmup:
                self.step_profile()

        batch.latents = video
        batch.audio_latents = audio
        return batch
