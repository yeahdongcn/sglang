"""End-to-end MLX model runner for Apple Silicon.

Runs the entire model within MLX, bypassing PyTorch MPS entirely.
"""

import logging
import time

import mlx.core as mx
from mlx_lm import load as mlx_lm_load

from sglang.srt.hardware_backend.mlx.kv_cache import (
    BatchedDecodeContext,
    ContiguousKVCache,
    MlxRequestState,
    OffsetCache,
    clear_context,
    extract_kv_cache,
    get_num_layers,
    patch_model_attention,
    set_context,
)

logger = logging.getLogger(__name__)


class MlxModelRunner:
    """Model runner that executes the entire model in MLX.

    This avoids the MPS<->MLX tensor bridge overhead by keeping all
    computation within MLX.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self._request_states: dict[str, MlxRequestState] = {}

        self._load_model()
        patch_model_attention(self.model)

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def prefill(
        self,
        req_id: str,
        token_ids: list[int],
    ) -> int:
        """Run prefill for a single request.

        If a request with the same req_id already has state (e.g. from a
        previous partial prefill), the existing KV cache is reused and only
        the new tokens are fed through the model.

        Args:
            req_id: Request identifier
            token_ids: Input token IDs (full sequence, including any
                previously prefilled tokens)

        Returns:
            Next token ID (greedy sampled)
        """
        existing_state = self._request_states.get(req_id)
        if existing_state is not None:
            cached_input_len = (
                len(existing_state.token_ids) - existing_state.generated_tokens
            )
            new_tokens = token_ids[cached_input_len:]
            cache = existing_state.cache
        else:
            new_tokens = token_ids
            num_layers = get_num_layers(self.model)
            cache = [ContiguousKVCache() for _ in range(num_layers)]

        input_ids = mx.array([new_tokens], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)

        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        mx.eval(next_token_mlx, *[c.state for c in cache])
        next_token = int(next_token_mlx.item())

        self._request_states[req_id] = MlxRequestState(
            token_ids=list(token_ids) + [next_token],
            cache=cache,
            generated_tokens=1,
        )

        return next_token

    def prefill_batch(
        self,
        req_ids: list[str],
        token_ids_list: list[list[int]],
    ) -> list[int]:
        """Run batched prefill for multiple requests in a single forward pass.

        When all sequences have the same length, they are stacked into a single
        batch tensor for one forward pass.  For variable-length sequences the
        method falls back to serial prefill.

        Args:
            req_ids: List of request identifiers
            token_ids_list: List of token ID sequences, one per request

        Returns:
            List of next token IDs (greedy sampled)
        """
        if len(req_ids) == 1:
            return [self.prefill(req_ids[0], token_ids_list[0])]

        # Check if all sequences have the same length (enables true batching)
        lengths = [len(tids) for tids in token_ids_list]
        if len(set(lengths)) != 1:
            # Variable lengths – fall back to serial prefill
            return [
                self.prefill(rid, tids) for rid, tids in zip(req_ids, token_ids_list)
            ]

        num_layers = get_num_layers(self.model)
        batch_cache = [ContiguousKVCache() for _ in range(num_layers)]

        batched_input = mx.array(
            [list(tids) for tids in token_ids_list], dtype=mx.int32
        )

        model_output = self.model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_tokens_mlx = mx.argmax(last_logits, axis=-1)

        mx.eval(next_tokens_mlx, *[c.state for c in batch_cache])
        next_tokens = next_tokens_mlx.tolist()

        for i, req_id in enumerate(req_ids):
            per_req_cache = extract_kv_cache(batch_cache, i)
            self._request_states[req_id] = MlxRequestState(
                token_ids=list(token_ids_list[i]) + [next_tokens[i]],
                cache=per_req_cache,
                generated_tokens=1,
            )

        return next_tokens

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Run decode for one or more requests.

        Uses the same SDPA wrapper path regardless of batch size.

        Args:
            req_ids: List of request IDs to decode

        Returns:
            List of next token IDs
        """
        states = [self._request_states[rid] for rid in req_ids]
        batch_size = len(states)

        num_layers = get_num_layers(self.model)
        layer_caches = [
            [state.cache[layer_idx] for state in states]
            for layer_idx in range(num_layers)
        ]
        seq_lens = [state.cache[0].offset for state in states]

        ctx = BatchedDecodeContext(
            batch_size=batch_size,
            seq_lens=seq_lens,
            layer_caches=layer_caches,
        )
        set_context(ctx)

        try:
            max_offset = max(seq_lens)
            shim_cache = [OffsetCache(offset=max_offset) for _ in range(num_layers)]

            last_tokens = [state.token_ids[-1] for state in states]
            batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

            model_output = self.model(batched_input, cache=shim_cache)
            logits = self._extract_logits(model_output)

            next_token_logits = logits[:, -1, :]
            next_tokens_mlx = mx.argmax(next_token_logits, axis=-1)

            eval_targets = [next_tokens_mlx]
            for state in states:
                for c in state.cache:
                    eval_targets.extend([c.keys, c.values])
            mx.eval(*eval_targets)

            next_tokens = next_tokens_mlx.tolist()

            for i, state in enumerate(states):
                state.token_ids.append(next_tokens[i])
                state.generated_tokens += 1

            return next_tokens
        finally:
            clear_context()

    def remove_request(self, req_id: str):
        """Clean up state for a completed request."""
        self._request_states.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._request_states.clear()
