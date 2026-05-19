"""Unit tests for server warmup request selection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = "ok"
        self._payload = payload

    def json(self):
        return self._payload


class TestServerWarmup(unittest.TestCase):
    def _server_args(self):
        return SimpleNamespace(
            api_key=None,
            debug_tensor_dump_input_file=None,
            disaggregation_mode="null",
            dp_size=1,
            skip_tokenizer_init=False,
            url=lambda: "http://127.0.0.1:1234",
            ssl_verify=lambda: False,
        )

    def _run_warmup(self, *, mlx_enabled: bool):
        model_info = {
            "has_image_understanding": True,
            "is_generation": True,
        }
        tokenizer_manager = SimpleNamespace(
            served_model_name="test-model",
            server_status=None,
        )
        global_state = SimpleNamespace(tokenizer_manager=tokenizer_manager)
        with (
            patch.object(http_server.time, "sleep"),
            patch.object(http_server, "use_mlx", return_value=mlx_enabled),
            patch.object(
                http_server,
                "_global_state",
                global_state,
            ),
            patch.object(
                http_server.requests,
                "get",
                return_value=_Response(model_info),
            ),
            patch.object(
                http_server.requests,
                "post",
                return_value=_Response({"ok": True}),
            ) as post,
        ):
            self.assertTrue(http_server._execute_server_warmup(self._server_args()))
        return post.call_args

    def test_mlx_vlm_model_uses_text_generation_warmup(self):
        call_args = self._run_warmup(mlx_enabled=True)

        self.assertEqual(call_args.args[0], "http://127.0.0.1:1234/generate")
        self.assertIn("text", call_args.kwargs["json"])
        self.assertNotIn("messages", call_args.kwargs["json"])

    def test_non_mlx_vlm_model_keeps_chat_warmup(self):
        call_args = self._run_warmup(mlx_enabled=False)

        self.assertEqual(call_args.args[0], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertIn("messages", call_args.kwargs["json"])
        self.assertNotIn("text", call_args.kwargs["json"])


if __name__ == "__main__":
    unittest.main()
