"""CPU coverage for ScheduleBatch prefill-chunk state propagation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_idle_schedule_batch(contains_last_prefill_chunk: bool) -> ScheduleBatch:
    req = SimpleNamespace(lora_id=None, rid="r0", token_type_ids=None)
    return ScheduleBatch(
        reqs=[req],
        device="cpu",
        forward_mode=ForwardMode.IDLE,
        input_ids=torch.empty(0, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([0], dtype=torch.int64),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        seq_lens_sum=0,
        contains_last_prefill_chunk=contains_last_prefill_chunk,
    )


class TestForwardBatchChunkState(unittest.TestCase):
    def test_default_is_conservative_and_init_new_copies_scheduler_state(self):
        direct = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
        )
        self.assertFalse(direct.contains_last_prefill_chunk)

        model_runner = SimpleNamespace(device=torch.device("cpu"))
        for scheduler_value in (False, True):
            with self.subTest(scheduler_value=scheduler_value):
                schedule_batch = _make_idle_schedule_batch(scheduler_value)
                self.assertIs(
                    schedule_batch.copy().contains_last_prefill_chunk,
                    scheduler_value,
                )
                with patch(
                    "sglang.srt.model_executor.forward_batch_info."
                    "enable_num_token_non_padded",
                    return_value=False,
                ):
                    forward_batch = ForwardBatch.init_new(
                        schedule_batch,
                        model_runner,
                        return_hidden_states_before_norm=False,
                    )
                self.assertIs(
                    forward_batch.contains_last_prefill_chunk, scheduler_value
                )


if __name__ == "__main__":
    unittest.main()
