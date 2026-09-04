# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.multimodal_gen.configs.pipeline_configs.magi2 import Magi2PipelineConfig
from sglang.multimodal_gen.configs.sample.magi2 import Magi2SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.stages import (
    Magi2DecodingStage,
    Magi2DenoisingStage,
    Magi2ImageEncodingStage,
    Magi2InputStage,
    Magi2LatentPreparationStage,
    Magi2PackingStage,
    Magi2StageHandoffStage,
    Magi2TextEncodingStage,
    Magi2TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.distributed import RankGenerator
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _magi2_expert_group_ranks(
    *,
    world_size: int,
    ep_size: int,
    cfg_parallel: bool,
    tp_size: int = 1,
    sp_size: int = 1,
    cfg_size: int = 1,
    dp_size: int = 1,
) -> list[list[int]]:
    """Return deterministic expert groups for a MAGI-2 parallel topology.

    In CFG2/SP4, an expert group must stay inside one sequence-parallel group
    (and hence one CFG branch).  Mixing ranks such as ``[0, 4]`` would make the
    MoE all-to-all exchange conditional and unconditional activations.  The
    legacy CFG1 layout intentionally retains its historical contiguous groups.
    """
    if ep_size < 1:
        raise ValueError(f"MAGI-2 ep_size must be positive, got {ep_size}")
    if world_size < 1 or world_size % ep_size:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"
        )
    if not cfg_parallel:
        return [
            list(range(start, start + ep_size))
            for start in range(0, world_size, ep_size)
        ]

    if cfg_size < 2:
        raise ValueError("CFG-parallel expert groups require cfg_size >= 2")
    if tp_size < 1 or sp_size < 1 or dp_size < 1:
        raise ValueError("tp_size, sp_size, and dp_size must be positive")
    expected_world = tp_size * sp_size * cfg_size * dp_size
    if expected_world != world_size:
        raise ValueError(
            "parallel dimensions do not match world_size: "
            f"tp={tp_size}, sp={sp_size}, cfg={cfg_size}, dp={dp_size} "
            f"=> {expected_world}, world_size={world_size}"
        )
    if sp_size % ep_size:
        raise ValueError(
            "CFG-parallel MAGI-2 expert groups must divide each SP group: "
            f"sp_size ({sp_size}) is not divisible by ep_size ({ep_size})"
        )

    # Use the same rank generator as initialize_model_parallel.  This remains
    # correct if a future TP/DP layout makes SP ranks non-contiguous globally.
    rank_generator = RankGenerator(
        tp_size,
        sp_size,
        1,
        cfg_size,
        dp_size,
        "tp-sp-pp-cfg-dp",
    )
    groups: list[list[int]] = []
    for sp_group_ranks in rank_generator.get_ranks("sp"):
        groups.extend(
            sp_group_ranks[start : start + ep_size]
            for start in range(0, sp_size, ep_size)
        )
    return groups


def _build_expert_parallel_group(server_args: ServerArgs):
    """``None`` for a single-rank deployment, so the MoE skips its collectives."""
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_dit_world_size,
    )

    if not dist.is_initialized():
        return None

    world_size = get_dit_world_size()
    cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
    cfg_size = int(
        getattr(server_args, "cfg_parallel_degree", None) or (2 if cfg_parallel else 1)
    )
    sp_size = int(
        getattr(server_args, "sp_degree", None)
        or (world_size // cfg_size if cfg_parallel else world_size)
    )
    tp_size = int(getattr(server_args, "tp_size", None) or 1)
    dp_size = int(getattr(server_args, "dp_size", None) or 1)
    # Match Magi2PipelineConfig's implicit degree: one EP group per CFG branch.
    default_ep_size = sp_size if cfg_parallel else world_size
    ep_size = server_args.pipeline_config.ep_size or default_ep_size
    if ep_size <= 1:
        return None

    groups = _magi2_expert_group_ranks(
        world_size=world_size,
        ep_size=ep_size,
        cfg_parallel=cfg_parallel,
        tp_size=tp_size,
        sp_size=sp_size,
        cfg_size=cfg_size,
        dp_size=dp_size,
    )
    rank = dist.get_rank()
    my_group = None
    # new_group is collective: all ranks enter every group, keep the one they are in.
    for ranks in groups:
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_group = group

    logger.info(
        "[magi2] expert parallel: world=%d cfg=%d sp=%d ep=%d groups=%s",
        world_size,
        cfg_size,
        sp_size,
        ep_size,
        groups,
    )
    return my_group


class Magi2Pipeline(ComposedPipelineBase):
    pipeline_name = "Magi2Pipeline"
    is_video_pipeline = True
    pipeline_config_cls = Magi2PipelineConfig
    sampling_params_cls = Magi2SamplingParams

    _required_config_modules = []

    def validate_disagg_role(self, role: RoleType) -> None:
        # add_stage silently drops role_affinity mismatches, yielding an empty pipeline.
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "Magi2Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict:
        """Bypasses the base, whose unconditional ``_load_config`` needs a ``model_index.json`` this checkpoint lacks."""
        from sglang.multimodal_gen.runtime.loader.component_loaders.magi2_loader import (
            load_magi2_modules,
        )

        if loaded_modules:
            return dict(loaded_modules)

        # Not in initialize_pipeline: that hook runs after load_modules, but the MoE
        # takes its process group at construction time.
        self.ep_group = _build_expert_parallel_group(server_args)
        return load_magi2_modules(
            server_args=server_args, pipeline=self, ep_group=self.ep_group
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        config = server_args.pipeline_config

        self.add_stage(Magi2InputStage())
        # Image before text: reversed, every image conditions on a zero vector.
        self.add_stage(
            Magi2ImageEncodingStage(
                vae=self.get_module("vae"),
                spatial_compression_ratio=(
                    config.vae_config.arch_config.spatial_compression_ratio
                ),
                latents_mean=config.vae_config.arch_config.latents_mean,
                latents_std=config.vae_config.arch_config.latents_std,
            )
        )
        self.add_stage(
            Magi2TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                tokenizer_kwargs=(
                    config.text_encoder_configs[0].arch_config.tokenizer_kwargs
                ),
                skip_layer=config.text_encoder_configs[0].arch_config.skip_layer,
            )
        )
        self.add_stage(Magi2LatentPreparationStage())
        self.add_stage(Magi2TimestepPreparationStage())

        self.add_stage(
            Magi2PackingStage(grid_key="magi2_preview_grid", conditions_on_images=True),
            stage_name="magi2_packing_preview",
        )
        self.add_stage(
            Magi2DenoisingStage(transformer=self.get_module("transformer")),
            stage_name="magi2_denoising_preview",
        )

        if config.enable_refiner:
            self.add_stage(Magi2StageHandoffStage())
            self.add_stage(
                Magi2PackingStage(grid_key="magi2_refiner_grid", refiner_only=True),
                stage_name="magi2_packing_refiner",
            )
            self.add_stage(
                Magi2DenoisingStage(
                    transformer=self.get_module("transformer_2"),
                    guidance_key="refiner",
                    refiner_only=True,
                ),
                stage_name="magi2_denoising_refiner",
            )

        vae_arch = config.vae_config.arch_config
        self.add_stage(
            Magi2DecodingStage(
                video_vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
                turbo_vae=self.get_module("turbo_vae"),
                latents_mean=vae_arch.latents_mean,
                latents_std=vae_arch.latents_std,
            )
        )


EntryClass = Magi2Pipeline
