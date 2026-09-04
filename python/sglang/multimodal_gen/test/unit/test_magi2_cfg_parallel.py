# SPDX-License-Identifier: Apache-2.0
"""Static and CPU-only contracts for the experimental MAGI-2 CFG2/SP4 path."""

from types import SimpleNamespace

import pytest
from sglang.multimodal_gen.configs.pipeline_configs.magi2 import Magi2PipelineConfig
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    _normalize_cfg_scales,
)
from sglang.multimodal_gen.runtime.pipelines.magi2 import _magi2_expert_group_ranks


def _server_args(**overrides):
    values = {
        "enable_cfg_parallel": True,
        "cfg_parallel_degree": 2,
        "num_gpus": 8,
        "tp_size": 1,
        "sp_degree": 4,
        "ulysses_degree": 4,
        "ring_degree": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_magi2_cfg_parallel_is_explicitly_supported_but_not_auto_enabled():
    deployment = Magi2PipelineConfig().get_model_deployment_config()

    assert deployment.supports_cfg_parallel
    assert not deployment.auto_enable_cfg_parallel


def test_magi2_cfg2_sp4_validation_uses_per_branch_sp_degree():
    config = Magi2PipelineConfig(enable_refiner=False)

    # The implicit EP degree is the four-rank SP branch, not the eight-rank
    # combined CFG world (which would fail the 12-head divisibility check).
    config.validate_server_args(_server_args())


def test_magi2_cfg_parallel_rejects_other_cfg_degrees():
    config = Magi2PipelineConfig(enable_refiner=False)

    with pytest.raises(ValueError, match="cfg_parallel_degree=2"):
        config.validate_server_args(_server_args(cfg_parallel_degree=4))


def test_magi2_cfg_parallel_requires_full_world_factorization():
    config = Magi2PipelineConfig(enable_refiner=False)

    with pytest.raises(ValueError, match="consume the full world"):
        config.validate_server_args(_server_args(num_gpus=4))


def test_magi2_cfg_parallel_rejects_cross_branch_expert_degree():
    config = Magi2PipelineConfig(enable_refiner=False, ep_size=8)

    with pytest.raises(ValueError, match=r"sp_degree \(4\).*ep_size \(8\)"):
        config.validate_server_args(_server_args())


def test_cfg1_sp8_validation_remains_available():
    config = Magi2PipelineConfig(enable_refiner=False, ep_size=4)

    config.validate_server_args(
        _server_args(
            enable_cfg_parallel=False,
            cfg_parallel_degree=1,
            sp_degree=8,
            ulysses_degree=8,
        )
    )


def test_cfg2_sp4_expert_groups_never_mix_branches():
    groups = _magi2_expert_group_ranks(
        world_size=8,
        ep_size=4,
        cfg_parallel=True,
        tp_size=1,
        sp_size=4,
        cfg_size=2,
        dp_size=1,
    )

    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_cfg_expert_groups_follow_strided_sp_order():
    # RankGenerator lays out TP before SP, so the SP groups are strided when
    # TP is enabled.  The helper must not fall back to globally contiguous
    # ranges (which would cross a TP lane and a CFG branch).
    groups = _magi2_expert_group_ranks(
        world_size=8,
        ep_size=2,
        cfg_parallel=True,
        tp_size=2,
        sp_size=2,
        cfg_size=2,
        dp_size=1,
    )

    assert groups == [[0, 2], [1, 3], [4, 6], [5, 7]]


def test_cfg1_expert_group_layout_is_unchanged():
    groups = _magi2_expert_group_ranks(
        world_size=8,
        ep_size=4,
        cfg_parallel=False,
    )

    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_cfg_scale_normalization_supports_joint_audio_video_outputs():
    assert _normalize_cfg_scales(5.0, 2) == (5.0, 5.0)
    assert _normalize_cfg_scales((5.0, 7.0), 2) == (5.0, 7.0)
    with pytest.raises(ValueError, match="match the number of model outputs"):
        _normalize_cfg_scales((5.0,), 2)
