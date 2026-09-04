# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.magi2 import (
    Magi2PreviewConfig,
    Magi2RefinerConfig,
)
from sglang.multimodal_gen.configs.models.encoders.magi2 import Magi2TextEncoderConfig
from sglang.multimodal_gen.configs.models.vaes.magi2 import (
    Magi2AudioVAEConfig,
    Magi2TurboVAEConfig,
    Magi2VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class Magi2PipelineConfig(PipelineConfig):
    """MAGI-2-preview: joint audio-video generation from one MoE checkpoint.

    A 40-layer MoE preview DiT and a 30-layer dense refiner DiT share the root
    checkpoint; the refiner is only reachable at 1080p.
    """

    task_type: ModelTaskType = ModelTaskType.TI2V

    # ref_image_type is "original", so the generic TI2V resize/centre-crop must not run.
    skip_input_image_preprocess: bool = True

    # No model_index.json, so nothing loads through sglang's diffusers loaders.
    native_only_components = (
        "text_encoder",
        "tokenizer",
        "transformer",
        "transformer_2",
        "vae",
        "turbo_vae",
        "audio_vae",
    )

    dit_config: Magi2PreviewConfig = field(default_factory=Magi2PreviewConfig)
    refiner_dit_config: Magi2RefinerConfig = field(default_factory=Magi2RefinerConfig)
    dit_precision: str = "bf16"

    vae_config: Magi2VideoVAEConfig = field(default_factory=Magi2VideoVAEConfig)
    vae_precision: str = "fp32"
    turbo_vae_config: Magi2TurboVAEConfig = field(default_factory=Magi2TurboVAEConfig)
    audio_vae_config: Magi2AudioVAEConfig = field(default_factory=Magi2AudioVAEConfig)
    audio_vae_precision: str = "fp32"

    text_encoder_configs: tuple[Magi2TextEncoderConfig, ...] = field(
        default_factory=lambda: (Magi2TextEncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{}])

    should_use_guidance: bool = True
    flow_shift: float | None = 7.0
    refiner_flow_shift: float = 5.0

    enable_refiner: bool = True

    # Index into the ZeroSNR discretization used to renoise the upsampled preview.
    refiner_renoise_index: int = 220

    use_turbo_vae: bool = True

    output_audio_sample_rate: int | None = 44100
    output_audio_channels: int | None = 2

    # In the legacy CFG1/SP8 layout this defaults to the world size, and callers
    # use ``ep_size=4`` to keep the 12-head router divisible.  In the experimental
    # CFG2/SP4 layout the default is the per-CFG SP group (4 ranks), so an
    # explicit pipeline-config override is not needed.
    ep_size: int | None = None

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            # CFG parallel is intentionally opt-in for MAGI-2.  The default
            # deployment remains CFG1/SP8 until the experimental path has been
            # validated on real hardware.
            supports_cfg_parallel=True,
            auto_enable_cfg_parallel=False,
            keep_resident_components=("vae", "turbo_vae", "audio_vae"),
        )

    def supports_disaggregation(self) -> bool:
        return False

    def _warn_about_allocator(self) -> None:
        """Warn at startup when the allocator will fail the 1080p decode."""
        if not self.enable_refiner:
            return
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in conf:
            return
        logger.warning(
            "[magi2] PYTORCH_CUDA_ALLOC_CONF does not enable expandable_segments "
            "(currently %r). The two-stage 1080p tier requests a single ~9 GiB "
            "block during decode while both DiTs are resident, which typically "
            "fails on fragmentation after several minutes of denoising. Set "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before launching, "
            "or run the preview-only tier.",
            conf or "unset",
        )

    def validate_server_args(self, server_args) -> None:
        super().validate_server_args(server_args)
        self._warn_about_allocator()

        cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
        cfg_degree = int(
            getattr(server_args, "cfg_parallel_degree", None)
            or (2 if cfg_parallel else 1)
        )
        num_gpus = int(getattr(server_args, "num_gpus", 1))
        tp_size = int(getattr(server_args, "tp_size", 1) or 1)
        dp_size = int(getattr(server_args, "dp_size", 1) or 1)
        sp_degree = int(
            getattr(server_args, "sp_degree", None)
            or (num_gpus // max(1, cfg_degree) if cfg_parallel else num_gpus)
        )
        ulysses_degree = int(
            getattr(server_args, "ulysses_degree", None)
            or (sp_degree if cfg_parallel else 1)
        )
        ring_degree = int(getattr(server_args, "ring_degree", 1) or 1)

        if cfg_parallel:
            # This branch is deliberately narrow: two CFG branches each run over
            # one SP4 replica.  Keeping the contract explicit prevents a partially
            # initialized model from silently mixing CFG and sequence groups.
            if cfg_degree != 2:
                raise ValueError(
                    "MAGI-2 experimental CFG parallelism currently supports only "
                    f"cfg_parallel_degree=2, got {cfg_degree}"
                )
            if tp_size != 1:
                raise ValueError(
                    "MAGI-2 CFG2/SP4 requires --tp-size 1; tensor parallelism "
                    f"({tp_size}) would create a second, unsupported weight axis"
                )
            if sp_degree != 4 or ulysses_degree != 4 or ring_degree != 1:
                raise ValueError(
                    "MAGI-2 experimental CFG parallelism requires "
                    "--sp-degree 4 --ulysses-degree 4 --ring-degree 1; got "
                    f"sp={sp_degree}, ulysses={ulysses_degree}, ring={ring_degree}"
                )
            expected_world = dp_size * tp_size * cfg_degree * sp_degree
            if num_gpus != expected_world:
                raise ValueError(
                    "MAGI-2 CFG2/SP4 parallel dimensions must consume the full "
                    f"world: dp={dp_size}, tp={tp_size}, cfg={cfg_degree}, "
                    f"sp={sp_degree} => {expected_world}, got num_gpus={num_gpus}"
                )

        if not cfg_parallel and tp_size > 1:
            raise ValueError(
                f"MAGI-2 has no tensor-parallel layers; --tp-size "
                f"({tp_size}) would leave those ranks duplicating "
                "work instead of sharding the sequence (measured 1.7x slower at "
                "tp=2, 4.9x at tp=4). Put the whole degree in --ulysses-degree."
            )

        if not cfg_parallel and ring_degree > 1:
            raise ValueError(
                "MAGI-2 shards its packed sequence over the full SP group but "
                "exchanges attention heads over the Ulysses group only; "
                f"--ring-degree ({ring_degree}) must be 1. Put the "
                "whole degree in --ulysses-degree."
            )

        # Expert groups must not cross CFG branches.  Therefore the implicit EP
        # degree is one per-CFG SP group in CFG2/SP4, while the legacy path keeps
        # its historical world-size default.
        ep_size = self.ep_size or (sp_degree if cfg_parallel else num_gpus)
        if ep_size < 1:
            raise ValueError(f"ep_size must be positive, got {ep_size}")
        ep_divisor = sp_degree if cfg_parallel else num_gpus
        if ep_divisor % ep_size:
            raise ValueError(
                f"{'sp_degree' if cfg_parallel else 'num_gpus'} ({ep_divisor}) "
                f"must be divisible by ep_size ({ep_size})"
            )

        moe_heads = self.dit_config.arch_config.moe_num_heads
        if moe_heads % ep_size:
            raise ValueError(
                f"MAGI-2 splits {moe_heads} MoE heads across expert-parallel "
                f"ranks; ep_size ({ep_size}) must divide {moe_heads}"
            )

        # Attention heads are sharded over each SP group in CFG mode.  Keep the
        # historical world-size check for CFG1 deployments (including any
        # unusual legacy DP layout) until that topology is separately audited.
        preview = self.dit_config.arch_config
        axes = {"preview attention heads": preview.num_attention_heads}
        if self.enable_refiner:
            refiner = self.refiner_dit_config.arch_config
            axes["refiner attention heads"] = refiner.num_attention_heads
            axes["refiner KV heads"] = refiner.num_query_groups

        attention_degree = sp_degree if cfg_parallel else num_gpus
        for name, count in sorted(axes.items(), key=lambda item: item[1]):
            if count % attention_degree:
                raise ValueError(
                    f"MAGI-2 shards {name} ({count}) across the SP group; "
                    f"degree ({attention_degree}) must divide it. Valid counts are "
                    f"{[n for n in range(1, max(axes.values()) + 1) if all(c % n == 0 for c in axes.values())]}"
                )
