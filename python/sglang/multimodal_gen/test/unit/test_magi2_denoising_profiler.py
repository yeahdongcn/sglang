# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.stages.denoising as magi2_denoising
import torch
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


class _Scheduler:
    def step(self, prediction, timestep, sample, return_dict=False):
        return (sample + prediction,)


class _SamplingParams:
    guidance_scale = 1.0
    audio_guidance_scale = 1.0
    skimmed_guidance_scale = None
    use_skimmed_guidance = False


class _CFGSamplingParams(_SamplingParams):
    guidance_scale = 5.0
    audio_guidance_scale = 7.0


class _PipelineConfig:
    should_use_guidance = True

    def __init__(self):
        from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy

        self.cfg_policy = CFGPolicy()

    def postprocess_cfg_noise(self, _batch, noise_pred, _noise_pred_cond):
        return noise_pred


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
    stage.step_profile = Mock()
    stage._predict = MethodType(
        lambda self, **kwargs: (torch.ones_like(kwargs["video"]), None),
        stage,
    )

    result = stage.forward(batch, server_args)

    assert len(batch.metrics.steps) == 2
    assert stage.step_profile.call_count == 2
    assert torch.equal(result.latents, torch.full_like(result.latents, 2))


def test_magi2_cfg2_runs_one_branch_per_cfg_rank(monkeypatch):
    """The stage delegates branch dispatch and keeps separate video/audio scales."""
    monkeypatch.setattr(
        magi2_denoising,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        magi2_denoising,
        "get_classifier_free_guidance_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        magi2_denoising.packed_sequence,
        "build_timesteps",
        lambda **kwargs: kwargs["video_t"],
    )

    calls = []
    layout = SimpleNamespace(
        ref_patch_index=torch.empty(0, dtype=torch.long), name="pos"
    )
    uncond_layout = SimpleNamespace(
        ref_patch_index=torch.empty(0, dtype=torch.long), name="neg"
    )
    batch = SimpleNamespace(
        is_warmup=False,
        perf_dump_path=None,
        metrics=RequestMetrics("magi2-cfg-test"),
        sampling_params=_CFGSamplingParams(),
        do_classifier_free_guidance=True,
        is_cfg_negative=False,
        cfg_normalization=0,
        guidance_rescale=0,
        extra={
            "magi2_layout": layout,
            "magi2_layout_uncond": uncond_layout,
            "magi2_coords": torch.empty(0),
            "magi2_coords_uncond": torch.empty(0),
        },
        prompt_embeds=[torch.tensor([1.0])],
        negative_prompt_embeds=[torch.tensor([-1.0])],
        latents=torch.zeros(1, 2),
        audio_latents=torch.zeros(1, 2),
        scheduler=_Scheduler(),
        timesteps=torch.tensor([1.0]),
    )
    batch.extra["magi2_audio_scheduler"] = _Scheduler()
    server_args = SimpleNamespace(
        enable_cfg_parallel=True,
        pipeline_config=_PipelineConfig(),
    )

    stage = object.__new__(magi2_denoising.Magi2DenoisingStage)
    stage.refiner_only = False
    stage.guidance_key = ""
    stage._current_use_nvtx = False
    stage.step_profile = lambda: None

    def fake_predict(self, **kwargs):
        text = float(kwargs["text"].item())
        calls.append((text, kwargs["layout"].name))
        value = text + 10.0 if batch.is_cfg_negative else text
        return torch.full_like(kwargs["video"], value), torch.full_like(
            kwargs["audio"], value
        )

    stage._predict = fake_predict.__get__(stage)

    def fake_cfg_dispatch(policy, predict_fn, cfg_scale, _batch, _pipeline_config):
        assert cfg_scale == (5.0, 7.0)
        # Simulate the two rank-local calls; the production utility calls only
        # one of these per process and combines them with a CFG all-reduce.
        cond = predict_fn(policy.branches[0])
        uncond = predict_fn(policy.branches[1])
        return tuple(
            scale * positive + (1.0 - scale) * negative
            for scale, positive, negative in zip(cfg_scale, cond, uncond)
        )

    monkeypatch.setattr(
        magi2_denoising, "run_two_branch_cfg_parallel", fake_cfg_dispatch
    )

    result = stage.forward(batch, server_args)

    assert calls == [(1.0, "pos"), (-1.0, "neg")]
    # The fake scheduler adds the combined prediction to the initial latent.
    assert torch.equal(result.latents, torch.full_like(result.latents, -31.0))
    assert torch.equal(
        result.audio_latents, torch.full_like(result.audio_latents, -47.0)
    )
    assert batch.is_cfg_negative is False
