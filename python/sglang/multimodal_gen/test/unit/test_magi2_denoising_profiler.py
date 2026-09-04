# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics
import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.stages.denoising as magi2_denoising


class _Scheduler:
    def step(self, prediction, timestep, sample, return_dict=False):
        return (sample + prediction,)


class _SamplingParams:
    guidance_scale = 1.0
    audio_guidance_scale = 1.0
    skimmed_guidance_scale = None
    use_skimmed_guidance = False


def test_magi2_denoising_records_each_step(monkeypatch):
    monkeypatch.setattr(
        magi2_denoising,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        magi2_denoising.packed_sequence,
        "build_timesteps",
        lambda **kwargs: kwargs["video_t"],
    )

    layout = SimpleNamespace(ref_patch_index=torch.empty(0, dtype=torch.long))
    batch = SimpleNamespace(
        is_warmup=False,
        perf_dump_path="/tmp/magi2-test-perf.json",
        metrics=RequestMetrics("magi2-profiler-test"),
        sampling_params=_SamplingParams(),
        extra={"magi2_layout": layout, "magi2_coords": torch.empty(0)},
        prompt_embeds=[torch.zeros(1)],
        negative_prompt_embeds=[],
        latents=torch.zeros(1, 2),
        audio_latents=None,
        scheduler=_Scheduler(),
        timesteps=torch.tensor([1.0, 2.0]),
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(should_use_guidance=False)
    )

    stage = object.__new__(magi2_denoising.Magi2DenoisingStage)
    stage.refiner_only = False
    stage.guidance_key = ""
    stage._current_use_nvtx = False
    stage._predict = MethodType(
        lambda self, **kwargs: (torch.ones_like(kwargs["video"]), None),
        stage,
    )

    result = stage.forward(batch, server_args)

    assert len(batch.metrics.steps) == 2
    assert torch.equal(result.latents, torch.full_like(result.latents, 2))
