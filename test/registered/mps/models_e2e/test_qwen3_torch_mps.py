"""Qwen3-0.6B cache contract for the per-operation MPS provider plan.

The production path uses the standard SRT ModelRunner and RadixCache.  MLX
borrows the Torch-owned model and pool only inside explicitly configured
operator islands.  The test runs whole-model MLX, Torch with Metal semantic
operators, mixed MPS/Torch-native attention, and the default all-Torch
numerical reference in separate, fully reaped server processes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
import torch
from safetensors.torch import save_file

from sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx import (
    QWEN3_COLD_PREFILL_MAX_TOKENS,
    QWEN3_COLD_PREFILL_MIN_TOKENS,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_mps_ci(est_time=480, suite="stage-b-e2e-mps")

MODEL_PATH = os.environ.get("SGLANG_MPS_TEST_MODEL", "Qwen/Qwen3-0.6B")
MEM_FRACTION_STATIC = os.environ.get("SGLANG_MPS_TEST_MEM_FRACTION", "0.6")
_MPS_PROVIDER_ENVS = (
    "SGLANG_MPS_QWEN3_MODEL_FORWARD",
    "SGLANG_MPS_QWEN3_GREEDY_TAIL",
    "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
    "SGLANG_MPS_QWEN3_RADIX_DECODE",
    "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
    "SGLANG_MPS_RMSNORM",
    "SGLANG_MPS_FUSED_ADD_RMSNORM",
    "SGLANG_MPS_SILU_AND_MUL",
)


def _create_tiny_qwen3_lora(adapter_dir: str) -> None:
    """Create a deterministic adapter without loading a second base model."""
    config = {
        "base_model_name_or_path": MODEL_PATH,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 2,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as config_file:
        json.dump(config, config_file)

    generator = torch.Generator(device="cpu").manual_seed(7)
    prefix = "base_model.model.model.layers.0.self_attn"
    tensors = {
        f"{prefix}.q_proj.lora_A.weight": (
            torch.randn(2, 1024, generator=generator, dtype=torch.bfloat16) * 0.02
        ),
        f"{prefix}.q_proj.lora_B.weight": (
            torch.randn(2048, 2, generator=generator, dtype=torch.bfloat16) * 0.02
        ),
        f"{prefix}.v_proj.lora_A.weight": (
            torch.randn(2, 1024, generator=generator, dtype=torch.bfloat16) * 0.02
        ),
        f"{prefix}.v_proj.lora_B.weight": (
            torch.randn(1024, 2, generator=generator, dtype=torch.bfloat16) * 0.02
        ),
    }
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
class TestQwen3MpsRadixCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _launch_server(self, *, profile: str):
        env = os.environ.copy()
        for name in _MPS_PROVIDER_ENVS:
            env.pop(name, None)
        if profile in {"mlx-model-forward", "mlx-model-forward-torch-tail"}:
            env.update(
                {
                    "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
                    "SGLANG_MPS_QWEN3_GREEDY_TAIL": (
                        "mlx,torch" if profile == "mlx-model-forward" else "torch"
                    ),
                    "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": ("metal_aot,metal_jit,torch"),
                    "SGLANG_MPS_QWEN3_RADIX_DECODE": ("metal_aot,metal_jit,torch"),
                    "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
                    "SGLANG_MPS_RMSNORM": "metal_jit,torch",
                    "SGLANG_MPS_FUSED_ADD_RMSNORM": "metal_jit,torch",
                    "SGLANG_MPS_SILU_AND_MUL": "metal_jit,torch",
                }
            )
        elif profile in {"torch-metal-ops", "torch-native-decode"}:
            env.update(
                {
                    "SGLANG_MPS_QWEN3_MODEL_FORWARD": "torch",
                    "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": ("metal_aot,metal_jit,torch"),
                    "SGLANG_MPS_QWEN3_RADIX_DECODE": ("metal_aot,metal_jit,torch"),
                    "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "torch",
                    "SGLANG_MPS_RMSNORM": "metal_jit,torch",
                    "SGLANG_MPS_FUSED_ADD_RMSNORM": "metal_jit,torch",
                    "SGLANG_MPS_SILU_AND_MUL": "metal_jit,torch",
                }
            )
        elif profile != "torch-reference":
            raise ValueError(f"unknown MPS E2E profile {profile!r}")
        env.pop("SGLANG_FORCE_FUSED_OP_BACKEND", None)

        other_args = [
            "--disable-overlap-schedule",
            "--sampling-backend",
            "pytorch",
            "--mem-fraction-static",
            MEM_FRACTION_STATIC,
            "--max-total-tokens",
            "4096",
            "--context-length",
            "2048",
        ]
        if profile == "torch-native-decode":
            other_args.extend(
                [
                    "--prefill-attention-backend",
                    "mps",
                    "--decode-attention-backend",
                    "torch_native",
                ]
            )

        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="mps",
            other_args=other_args,
            env=env,
        )

    def _launch_lora_server(self, adapter_dir: str):
        env = os.environ.copy()
        for name in _MPS_PROVIDER_ENVS:
            env.pop(name, None)
        env["SGLANG_MPS_QWEN3_MODEL_FORWARD"] = "mlx,torch"
        env.pop("SGLANG_FORCE_FUSED_OP_BACKEND", None)
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="mps",
            env=env,
            other_args=[
                "--disable-overlap-schedule",
                "--sampling-backend",
                "pytorch",
                "--mem-fraction-static",
                MEM_FRACTION_STATIC,
                "--max-total-tokens",
                "4096",
                "--context-length",
                "2048",
                "--enable-lora",
                "--lora-paths",
                adapter_dir,
                "--lora-backend",
                "torch_native",
                "--max-lora-rank",
                "2",
                "--max-loras-per-batch",
                "2",
            ],
        )

    def _generate_request(
        self,
        input_payload: dict[str, Any],
        *,
        return_logprob: bool = True,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                **input_payload,
                "sampling_params": {
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "ignore_eos": True,
                    "max_new_tokens": 4,
                    **(sampling_overrides or {}),
                },
                "return_logprob": return_logprob,
                "return_text_in_logprobs": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _generate(self, prompt: str, *, return_logprob: bool = True) -> dict:
        return self._generate_request({"text": prompt}, return_logprob=return_logprob)

    def _generate_without_logprobs(self, prompt: str) -> dict:
        return self._generate(prompt, return_logprob=False)

    def _flush_cache(self) -> None:
        response = requests.post(f"{self.base_url}/flush_cache", timeout=30)
        response.raise_for_status()

    def _mps_operator_state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        response.raise_for_status()
        states = [
            state["mps_operator"]
            for state in response.json()["internal_states"]
            if "mps_operator" in state
        ]
        self.assertEqual(len(states), 1)
        return states[0]

    def _reload_weights_from_disk(self) -> None:
        response = requests.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": self.model, "flush_cache": True},
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        self.assertTrue(result["success"], result["message"])

    def _exercise_radix_cache(
        self,
        *,
        profile: str,
    ) -> dict[str, Any]:
        process = self._launch_server(profile=profile)
        try:
            return self._assert_radix_cache_contract(profile)
        finally:
            kill_process_tree(process.pid, wait_timeout=30)
            # kill_process_tree waits for the OS process tree, but the local
            # Popen handle still needs to be reaped by this test process.
            process.wait(timeout=5)

    def _assert_radix_cache_contract(self, profile: str) -> dict[str, Any]:
        expect_mlx_greedy = profile == "mlx-model-forward"
        expect_mlx_torch_tail = profile == "mlx-model-forward-torch-tail"
        expect_mlx = expect_mlx_greedy or expect_mlx_torch_tail
        expect_metal_ops = profile == "torch-metal-ops"
        expect_metal_qkv_only = profile == "torch-native-decode"
        prefix = (
            "This document describes computers, mathematics, geography, and "
            "science in clear English. " * 36
        )
        target_prompt = prefix + "The capital of France is"

        self._flush_cache()
        cold_prefill_before = (
            self._mps_operator_state()["whole_model_prefill_call_count"]
            if expect_mlx
            else None
        )
        cold = self._generate(target_prompt)
        cold_prefill_after = (
            self._mps_operator_state()["whole_model_prefill_call_count"]
            if expect_mlx
            else None
        )

        self._flush_cache()
        self._generate(prefix + "The capital of Germany is")
        primed_prefill_count = (
            self._mps_operator_state()["whole_model_prefill_call_count"]
            if expect_mlx
            else None
        )
        warm = self._generate(target_prompt)
        warm_prefill_count = (
            self._mps_operator_state()["whole_model_prefill_call_count"]
            if expect_mlx
            else None
        )
        repeated = self._generate(target_prompt)
        repeated_prefill_count = (
            self._mps_operator_state()["whole_model_prefill_call_count"]
            if expect_mlx
            else None
        )

        cold_cached = cold["meta_info"]["cached_tokens"]
        shared_cached = warm["meta_info"]["cached_tokens"]
        repeated_cached = repeated["meta_info"]["cached_tokens"]
        prompt_tokens = repeated["meta_info"]["prompt_tokens"]

        self.assertEqual(cold_cached, 0)
        self.assertGreater(shared_cached, 0)
        self.assertLess(shared_cached, prompt_tokens)
        self.assertGreater(repeated_cached, shared_cached)
        self.assertGreaterEqual(
            repeated_cached,
            prompt_tokens - 2,
        )
        if expect_mlx:
            self.assertGreaterEqual(prompt_tokens, QWEN3_COLD_PREFILL_MIN_TOKENS)
            self.assertLessEqual(prompt_tokens, QWEN3_COLD_PREFILL_MAX_TOKENS)
            self.assertGreater(cold_prefill_after, cold_prefill_before)
            self.assertGreater(primed_prefill_count, cold_prefill_after)
            # Both target requests now have a Radix prefix. They must stay on
            # the standard Torch path instead of pretending to be a complete
            # prefix-free causal sequence inside the MLX island.
            self.assertEqual(warm_prefill_count, primed_prefill_count)
            self.assertEqual(repeated_prefill_count, primed_prefill_count)
        self.assertEqual(cold["output_ids"], warm["output_ids"])
        self.assertEqual(warm["output_ids"], repeated["output_ids"])

        cold_logprobs = [x[0] for x in cold["meta_info"]["output_token_logprobs"]]
        warm_logprobs = [x[0] for x in warm["meta_info"]["output_token_logprobs"]]
        self.assertLess(
            max(abs(a - b) for a, b in zip(cold_logprobs, warm_logprobs)), 0.15
        )

        server_info = requests.get(f"{self.base_url}/server_info", timeout=30)
        server_info.raise_for_status()
        mps_operator_states = [
            state["mps_operator"]
            for state in server_info.json()["internal_states"]
            if "mps_operator" in state
        ]
        self.assertEqual(len(mps_operator_states), 1)
        operator_state = mps_operator_states[0]
        if expect_mlx:
            # Prefix-free cold prefill and exact decode use whole-model MLX
            # islands. Prefix-hit prefill continues through Torch/Radix.
            self.assertTrue(operator_state["enabled"])
            self.assertGreater(operator_state["patched_attention_modules"], 0)
            self.assertGreater(operator_state["attention_qkv_call_count"], 0)
            self.assertEqual(
                operator_state["provider_priorities"]["model_forward"][0],
                "mlx",
            )
            self.assertEqual(operator_state["attention_backend"], "mps")
            self.assertIn(
                operator_state["qkv_kernel_backend"],
                {"metal_aot", "metal_jit"},
            )
            self.assertIn(
                operator_state["decode_kernel_backend"],
                {"metal_aot", "metal_jit"},
            )
            self.assertEqual(
                operator_state["deferred_kv_commit_backend"],
                "metal_jit",
            )
            self.assertEqual(operator_state["attention_decode_call_count"], 0)
            self.assertEqual(operator_state["attention_qkv_fallback_count"], 0)
            self.assertEqual(operator_state["attention_decode_fallback_count"], 0)
            self.assertEqual(operator_state["whole_model_backend"], "mlx")
            self.assertGreater(operator_state["whole_model_decode_call_count"], 0)
            self.assertTrue(operator_state["whole_model_compile_total_enabled"])
            self.assertEqual(
                operator_state["whole_model_compile_total_warmup_count"], 1
            )
            if expect_mlx_greedy:
                # The greedy gate selects the one resident 28-layer compiled
                # graph. Logprob requests use the eager hidden-output fallback
                # rather than retaining a second compiled model on 16 GB Macs.
                self.assertFalse(operator_state["whole_model_compile_enabled"])
                self.assertEqual(
                    operator_state["whole_model_compile_primary_variant"], "greedy"
                )
                self.assertEqual(operator_state["whole_model_compile_warmup_count"], 0)
                self.assertEqual(operator_state["whole_model_compile_call_count"], 0)
                self.assertTrue(operator_state["whole_model_greedy_tail_enabled"])
                self.assertEqual(
                    operator_state["whole_model_greedy_tail_backend"], "mlx"
                )
                self.assertTrue(operator_state["whole_model_greedy_compile_enabled"])
                self.assertEqual(
                    operator_state["whole_model_greedy_compile_warmup_count"], 1
                )
            else:
                self.assertTrue(operator_state["whole_model_compile_enabled"])
                self.assertEqual(
                    operator_state["whole_model_compile_primary_variant"], "hidden"
                )
                self.assertEqual(operator_state["whole_model_compile_warmup_count"], 1)
                self.assertGreater(operator_state["whole_model_compile_call_count"], 0)
                self.assertFalse(operator_state["whole_model_greedy_tail_enabled"])
                self.assertEqual(
                    operator_state["whole_model_greedy_tail_backend"], "torch"
                )
                self.assertFalse(operator_state["whole_model_greedy_compile_enabled"])
            self.assertEqual(
                operator_state["whole_model_prefill_call_count"],
                primed_prefill_count,
            )

            # This request is the exact gate A/B: both profiles use the same
            # whole-model MLX island, while only the selected tail changes.
            self._flush_cache()
            greedy_before = self._mps_operator_state()
            greedy = self._generate(target_prompt, return_logprob=False)
            greedy_after = self._mps_operator_state()
            self.assertEqual(greedy["output_ids"], cold["output_ids"])
            self.assertEqual(len(greedy["output_ids"]), 4)
            self.assertEqual(
                greedy_after["whole_model_greedy_tail_fallback_count"]
                - greedy_before["whole_model_greedy_tail_fallback_count"],
                0,
            )
            self.assertEqual(
                greedy_after["whole_model_call_count"]
                - greedy_before["whole_model_call_count"],
                4,
            )
            self.assertEqual(
                greedy_after["whole_model_prefill_call_count"]
                - greedy_before["whole_model_prefill_call_count"],
                1,
            )
            self.assertEqual(
                greedy_after["whole_model_decode_call_count"]
                - greedy_before["whole_model_decode_call_count"],
                3,
            )
            self.assertEqual(
                greedy_after["whole_model_compile_total_call_count"]
                - greedy_before["whole_model_compile_total_call_count"],
                3,
            )
            if expect_mlx_greedy:
                # No hidden state crosses back for a second LM-head projection.
                self.assertEqual(
                    greedy_after["whole_model_greedy_tail_call_count"]
                    - greedy_before["whole_model_greedy_tail_call_count"],
                    4,
                )
                self.assertEqual(
                    greedy_after["whole_model_greedy_tail_torch_call_count"]
                    - greedy_before["whole_model_greedy_tail_torch_call_count"],
                    0,
                )
                self.assertEqual(
                    greedy_after["whole_model_greedy_compile_call_count"]
                    - greedy_before["whole_model_greedy_compile_call_count"],
                    3,
                )
                self.assertEqual(
                    greedy_after["whole_model_compile_call_count"]
                    - greedy_before["whole_model_compile_call_count"],
                    0,
                )

                # Grammar is a standard SGLang feature, not a reason to leave
                # the whole MLX transformer island. Only the precomputed argmax
                # tail must fall back to Torch logits + xgrammar masking.
                grammar_before = self._mps_operator_state()
                constrained = self._generate_request(
                    {"text": "Output only lowercase letters:"},
                    return_logprob=False,
                    sampling_overrides={"regex": "[a-z ]+"},
                )
                grammar_after = self._mps_operator_state()
                self.assertIsNotNone(re.fullmatch(r"[a-z ]+", constrained["text"]))
                grammar_whole_model_delta = (
                    grammar_after["whole_model_call_count"]
                    - grammar_before["whole_model_call_count"]
                )
                self.assertGreater(grammar_whole_model_delta, 0)
                self.assertEqual(
                    grammar_after["whole_model_greedy_tail_fallback_count"]
                    - grammar_before["whole_model_greedy_tail_fallback_count"],
                    grammar_whole_model_delta,
                )
            else:
                self.assertEqual(
                    greedy_after["whole_model_greedy_tail_call_count"]
                    - greedy_before["whole_model_greedy_tail_call_count"],
                    0,
                )
                self.assertEqual(
                    greedy_after["whole_model_greedy_tail_torch_call_count"]
                    - greedy_before["whole_model_greedy_tail_torch_call_count"],
                    4,
                )
                self.assertEqual(
                    greedy_after["whole_model_compile_call_count"]
                    - greedy_before["whole_model_compile_call_count"],
                    3,
                )
                self.assertEqual(
                    greedy_after["whole_model_greedy_compile_call_count"]
                    - greedy_before["whole_model_greedy_compile_call_count"],
                    0,
                )

            # Default disk reload mutates the Torch Parameters in place.  The
            # next whole-model MLX prefill must rebuild its borrowed views and remain
            # numerically equivalent instead of retaining stale weight state.
            prefill_count_before_reload = greedy_after["whole_model_prefill_call_count"]
            self._reload_weights_from_disk()
            reloaded = self._generate(target_prompt)
            self.assertEqual(reloaded["output_ids"], cold["output_ids"])
            reloaded_logprobs = [
                x[0] for x in reloaded["meta_info"]["output_token_logprobs"]
            ]
            self.assertLess(
                max(abs(a - b) for a, b in zip(cold_logprobs, reloaded_logprobs)),
                0.15,
            )
            reloaded_server_info = requests.get(
                f"{self.base_url}/server_info", timeout=30
            )
            reloaded_server_info.raise_for_status()
            reloaded_operator_states = [
                state["mps_operator"]
                for state in reloaded_server_info.json()["internal_states"]
                if "mps_operator" in state
            ]
            self.assertEqual(len(reloaded_operator_states), 1)
            self.assertGreater(
                reloaded_operator_states[0]["whole_model_prefill_call_count"],
                prefill_count_before_reload,
            )

            # A second prefix-free request proves that the selected compiled
            # graph observes the refreshed capture list after a disk reload.
            self._flush_cache()
            reloaded_greedy_before = self._mps_operator_state()
            reloaded_greedy = self._generate(target_prompt, return_logprob=False)
            reloaded_greedy_after = self._mps_operator_state()
            self.assertEqual(reloaded_greedy["output_ids"], cold["output_ids"])
            tail_counter = (
                "whole_model_greedy_tail_call_count"
                if expect_mlx_greedy
                else "whole_model_greedy_tail_torch_call_count"
            )
            compile_counter = (
                "whole_model_greedy_compile_call_count"
                if expect_mlx_greedy
                else "whole_model_compile_call_count"
            )
            self.assertEqual(
                reloaded_greedy_after[tail_counter]
                - reloaded_greedy_before[tail_counter],
                4,
            )
            self.assertEqual(
                reloaded_greedy_after[compile_counter]
                - reloaded_greedy_before[compile_counter],
                3,
            )
        elif expect_metal_ops:
            # This is the core Torch-owned path: standard ModelRunner forward
            # with independently pinned Metal QKV preparation and Radix decode.
            # Both counters must move, otherwise an apparently selected provider
            # was bypassed by the real scheduler/model path.
            self.assertTrue(operator_state["enabled"])
            self.assertEqual(
                operator_state["provider_priorities"]["model_forward"],
                ["torch"],
            )
            self.assertEqual(operator_state["attention_backend"], "mps")
            self.assertGreater(operator_state["patched_qkv_modules"], 0)
            self.assertGreater(operator_state["patched_decode_modules"], 0)
            self.assertIn(
                operator_state["qkv_kernel_backend"],
                {"metal_aot", "metal_jit"},
            )
            self.assertIn(
                operator_state["decode_kernel_backend"],
                {"metal_aot", "metal_jit"},
            )
            self.assertGreater(operator_state["attention_qkv_call_count"], 0)
            self.assertGreater(operator_state["attention_decode_call_count"], 0)
            self.assertEqual(operator_state["attention_qkv_fallback_count"], 0)
            self.assertEqual(operator_state["attention_decode_fallback_count"], 0)
            self.assertEqual(operator_state["deferred_kv_commit_backend"], "off")
            self.assertEqual(operator_state["whole_model_backend"], "torch")
            self.assertEqual(operator_state["whole_model_call_count"], 0)
            self.assertFalse(operator_state["whole_model_compile_enabled"])
            self.assertEqual(
                operator_state["generic_kernel_backends"],
                {
                    "rmsnorm": "metal_jit",
                    "fused_add_rmsnorm": "metal_jit",
                    "silu_and_mul": "metal_jit",
                },
            )
        elif expect_metal_qkv_only:
            # QKV preparation is model-owned and remains eligible, while the
            # explicit Torch-native decode backend cannot consume the Radix
            # decode provider and must report its startup-time demotion.
            self.assertTrue(operator_state["enabled"])
            self.assertEqual(
                operator_state["provider_priorities"]["model_forward"],
                ["torch"],
            )
            self.assertEqual(
                operator_state["attention_backend"],
                "prefill=mps,decode=torch_native",
            )
            self.assertGreater(operator_state["patched_qkv_modules"], 0)
            self.assertEqual(operator_state["patched_decode_modules"], 0)
            self.assertIn(
                operator_state["qkv_kernel_backend"],
                {"metal_aot", "metal_jit"},
            )
            self.assertEqual(operator_state["decode_kernel_backend"], "torch")
            self.assertGreater(operator_state["attention_qkv_call_count"], 0)
            self.assertEqual(operator_state["attention_decode_call_count"], 0)
            self.assertEqual(operator_state["attention_qkv_fallback_count"], 0)
            self.assertEqual(operator_state["attention_decode_fallback_count"], 0)
            self.assertIn(
                "does not consume",
                operator_state["decode_fallback_reason"],
            )
            self.assertEqual(operator_state["whole_model_backend"], "torch")
            self.assertEqual(
                operator_state["generic_kernel_backends"],
                {
                    "rmsnorm": "metal_jit",
                    "fused_add_rmsnorm": "metal_jit",
                    "silu_and_mul": "metal_jit",
                },
            )
        else:
            self.assertFalse(operator_state["enabled"])
            self.assertEqual(
                operator_state["provider_priorities"]["model_forward"],
                ["torch"],
            )
            self.assertEqual(operator_state["attention_backend"], "mps")
            self.assertEqual(operator_state["qkv_kernel_backend"], "torch")
            self.assertEqual(operator_state["decode_kernel_backend"], "torch")
            self.assertEqual(operator_state["deferred_kv_commit_backend"], "off")
            self.assertEqual(operator_state["attention_qkv_call_count"], 0)
            self.assertEqual(operator_state["attention_decode_call_count"], 0)
            self.assertEqual(operator_state["attention_qkv_fallback_count"], 0)
            self.assertEqual(operator_state["attention_decode_fallback_count"], 0)
            self.assertEqual(operator_state["whole_model_backend"], "torch")
            self.assertEqual(
                operator_state["generic_kernel_backends"],
                {
                    "rmsnorm": "torch",
                    "fused_add_rmsnorm": "torch",
                    "silu_and_mul": "torch",
                },
            )
        # Exercise the actual HTTP/scheduler concurrency contract, rather than
        # only issuing serial cache probes.  Four long prompts fit within the
        # configured 4096-token pool and exercise the real scheduler contract.
        concurrent_prompts = [
            prefix + question
            for question in (
                "The capital of France is",
                "The capital of Germany is",
                "The capital of Italy is",
                "The capital of Spain is",
            )
        ]
        self._flush_cache()
        concurrent_state_before = self._mps_operator_state() if expect_mlx else None
        with ThreadPoolExecutor(max_workers=len(concurrent_prompts)) as executor:
            concurrent_results = list(
                executor.map(self._generate_without_logprobs, concurrent_prompts)
            )
        self.assertEqual(len(concurrent_results), len(concurrent_prompts))
        for result in concurrent_results:
            self.assertEqual(len(result["output_ids"]), 4)
        if expect_mlx:
            concurrent_state_after = self._mps_operator_state()
            whole_model_delta = (
                concurrent_state_after["whole_model_call_count"]
                - concurrent_state_before["whole_model_call_count"]
            )
            selected_tail_counter = (
                "whole_model_greedy_tail_call_count"
                if expect_mlx_greedy
                else "whole_model_greedy_tail_torch_call_count"
            )
            unselected_tail_counter = (
                "whole_model_greedy_tail_torch_call_count"
                if expect_mlx_greedy
                else "whole_model_greedy_tail_call_count"
            )
            self.assertGreater(whole_model_delta, 0)
            self.assertEqual(
                concurrent_state_after[selected_tail_counter]
                - concurrent_state_before[selected_tail_counter],
                whole_model_delta,
            )
            self.assertEqual(
                concurrent_state_after[unselected_tail_counter]
                - concurrent_state_before[unselected_tail_counter],
                0,
            )
            self.assertEqual(
                concurrent_state_after["whole_model_greedy_tail_fallback_count"],
                concurrent_state_before["whole_model_greedy_tail_fallback_count"],
            )
            self.assertGreaterEqual(
                concurrent_state_after["whole_model_max_decode_batch_size"],
                2,
            )

        return {"cold": cold, "concurrent": concurrent_results}

    def test_mlx_and_metal_profiles_match_default_torch_reference(self):
        results = {}
        profiles = (
            "mlx-model-forward",
            "mlx-model-forward-torch-tail",
            "torch-metal-ops",
            "torch-native-decode",
            "torch-reference",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                results[profile] = self._exercise_radix_cache(profile=profile)

        # Explicitly configured providers must retain the deterministic
        # behavior of the default all-Torch reference.
        if len(results) == len(profiles):
            reference = results["torch-reference"]
            reference_logprobs = [
                item[0]
                for item in reference["cold"]["meta_info"]["output_token_logprobs"]
            ]
            for profile in (
                "mlx-model-forward",
                "mlx-model-forward-torch-tail",
                "torch-metal-ops",
                "torch-native-decode",
            ):
                candidate = results[profile]
                with self.subTest(reference_comparison=profile):
                    self.assertEqual(
                        reference["cold"]["output_ids"],
                        candidate["cold"]["output_ids"],
                    )
                    candidate_logprobs = [
                        item[0]
                        for item in candidate["cold"]["meta_info"][
                            "output_token_logprobs"
                        ]
                    ]
                    self.assertLess(
                        max(
                            abs(reference_value - candidate_value)
                            for reference_value, candidate_value in zip(
                                reference_logprobs, candidate_logprobs
                            )
                        ),
                        0.15,
                    )
                    self.assertEqual(
                        [result["output_ids"] for result in reference["concurrent"]],
                        [result["output_ids"] for result in candidate["concurrent"]],
                    )

    def test_torch_native_lora_reuses_the_standard_model_runner(self):
        with tempfile.TemporaryDirectory(prefix="sglang_mps_qwen3_lora_") as adapter:
            _create_tiny_qwen3_lora(adapter)
            process = self._launch_lora_server(adapter)
            try:
                state = self._mps_operator_state()
                self.assertEqual(state["whole_model_backend"], "torch")
                self.assertIn("enable_lora", state["whole_model_fallback_reason"])

                base = self._generate("The capital of France is")
                adapted = self._generate_request(
                    {
                        "text": "The capital of France is",
                        "lora_path": adapter,
                    }
                )
                base_logprobs = [
                    item[0] for item in base["meta_info"]["output_token_logprobs"]
                ]
                adapted_logprobs = [
                    item[0] for item in adapted["meta_info"]["output_token_logprobs"]
                ]
                self.assertGreater(
                    max(
                        abs(base_value - adapted_value)
                        for base_value, adapted_value in zip(
                            base_logprobs, adapted_logprobs
                        )
                    ),
                    1e-4,
                )
            finally:
                kill_process_tree(process.pid, wait_timeout=30)
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
