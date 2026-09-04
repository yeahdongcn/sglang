# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    Magi2PostAdapter,
)


def test_local_post_adapter_matches_gather_then_project():
    torch.manual_seed(7)
    adapter = Magi2PostAdapter(
        SimpleNamespace(
            residual_stream_dim=8,
            video_in_channels=3,
            audio_in_channels=2,
        )
    )
    rows = torch.randn(10, 8)
    layout = SimpleNamespace(
        video_index=torch.tensor([0, 3, 4, 8]),
        audio_index=torch.tensor([1, 5]),
    )

    expected_video, expected_audio = adapter(rows, layout=layout)
    video_chunks = []
    audio_chunks = []
    for rank in range(3):
        start = rank * 4
        local = rows[start : start + 4]
        if local.shape[0] < 4:
            local = torch.cat((local, torch.zeros(4 - local.shape[0], 8)))
        plan = SimpleNamespace(sp_rank=rank, sp_size=3, local_len=4, orig_len=10)
        video, audio = adapter.forward_local(local, layout=layout, plan=plan)
        video_chunks.append(video)
        audio_chunks.append(audio)

    video_rows = torch.cat(video_chunks)[:10]
    audio_rows = torch.cat(audio_chunks)[:10]
    actual_video = video_rows.index_select(0, layout.video_index)
    actual_audio = audio_rows.index_select(0, layout.audio_index)

    torch.testing.assert_close(actual_video, expected_video)
    torch.testing.assert_close(actual_audio, expected_audio)
