"""MLX overlap scheduling mixin for the SGLang scheduler.

Provides ``event_loop_overlap_mlx`` and ``_run_batch_mlx_overlap`` which
are structurally identical to the normal event loop but force in-place
tensor operations in ``prepare_for_decode``.

Non-in-place MPS tensor allocations (e.g. ``seq_lens + 1`` instead of
``seq_lens.add_(1)``) create fresh Metal buffers each decode step.
These interfere with the MLX Metal command stream, adding ~8 ms per
decode step.  Forcing in-place ops eliminates this overhead and brings
overlap ON performance to parity with overlap OFF.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


class SchedulerMlxOverlapMixin:
    """Mixin that adds MLX overlap scheduling to :class:`Scheduler`."""

    def _run_batch_mlx_overlap(
        self: "Scheduler",
        worker_batch: "ModelWorkerBatch",
    ) -> GenerationBatchResult:
        """Run forward via the MLX path (same as non-overlap)."""
        self.tp_worker._ensure_mlx_pool_initialized()
        return self.tp_worker._forward_batch_generation_mlx(worker_batch)

    @DynamicGradMode()
    def event_loop_overlap_mlx(self: "Scheduler"):
        """Scheduler loop optimised for MLX on Apple Silicon.

        Structurally identical to ``event_loop_normal`` but forces
        in-place tensor operations in ``prepare_for_decode`` to avoid
        MPS tensor allocation overhead that interferes with MLX Metal
        command streams.  Forward passes are routed through
        ``_run_batch_mlx_overlap`` which calls ``decode_batch`` directly
        (no ``torch.tensor`` round-trip).
        """

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            # Force in-place ops in prepare_for_decode for the running
            # batch.  Non-in-place MPS tensor allocations (seq_lens + 1
            # instead of seq_lens.add_(1)) create new Metal buffers each
            # step which compete with the MLX Metal stream.
            if self.running_batch is not None:
                self.running_batch.enable_overlap = False

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.cancel_bubble_timer()
                self.self_check_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()
