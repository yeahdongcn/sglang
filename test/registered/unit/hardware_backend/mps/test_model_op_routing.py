"""CPU-safe tests for model-neutral MPS operator routing and lifecycle."""

import types
import unittest
from unittest import mock

from sglang.srt.hardware_backend.mps.model_ops.registry import (
    MpsModelOperatorRegistry,
    MpsModelOperatorSpec,
    model_architectures,
)
from sglang.srt.hardware_backend.mps.model_ops.router import (
    install_mps_model_operators,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _model_config(*architectures):
    return types.SimpleNamespace(
        hf_text_config=types.SimpleNamespace(architectures=list(architectures)),
        hf_config=types.SimpleNamespace(architectures=list(architectures)),
    )


class TestMpsModelOperatorRegistry(unittest.TestCase):
    def test_architectures_are_unique_and_ordered(self):
        config = types.SimpleNamespace(
            hf_text_config={"architectures": ["Qwen3ForCausalLM", "Shared"]},
            hf_config=types.SimpleNamespace(
                architectures=["Shared", "OtherForCausalLM"]
            ),
        )
        self.assertEqual(
            model_architectures(config),
            ("Qwen3ForCausalLM", "Shared", "OtherForCausalLM"),
        )

    def test_registry_rejects_duplicate_names_and_architectures(self):
        first = MpsModelOperatorSpec(
            name="first",
            architectures=frozenset({"ModelA"}),
            installer_path="package.first:install",
        )
        registry = MpsModelOperatorRegistry([first])
        with self.assertRaisesRegex(ValueError, "duplicate.*name"):
            registry.register(
                MpsModelOperatorSpec(
                    name="first",
                    architectures=frozenset({"ModelB"}),
                    installer_path="package.second:install",
                )
            )
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(
                MpsModelOperatorSpec(
                    name="second",
                    architectures=frozenset({"ModelA"}),
                    installer_path="package.second:install",
                )
            )

    def test_resolve_does_not_import_model_module(self):
        spec = MpsModelOperatorSpec(
            name="qwen3",
            architectures=frozenset({"Qwen3ForCausalLM"}),
            installer_path="package.qwen3:install",
        )
        registry = MpsModelOperatorRegistry([spec])
        with mock.patch("importlib.import_module") as import_module:
            self.assertIs(registry.resolve(_model_config("Qwen3ForCausalLM")), spec)
            self.assertIsNone(registry.resolve(_model_config("LlamaForCausalLM")))
        import_module.assert_not_called()


class TestMpsModelOperatorRouter(unittest.TestCase):
    def test_unknown_model_is_a_noop_without_lazy_import(self):
        registry = MpsModelOperatorRegistry()
        with mock.patch("importlib.import_module") as import_module:
            plan = install_mps_model_operators(
                object(),
                _model_config("UnknownForCausalLM"),
                object(),
                req_to_token_pool=object(),
                token_to_kv_pool=object(),
                registry=registry,
            )
        self.assertIsNone(plan)
        import_module.assert_not_called()

    def test_matching_model_loads_and_invokes_only_its_installer(self):
        spec = MpsModelOperatorSpec(
            name="qwen3",
            architectures=frozenset({"Qwen3ForCausalLM"}),
            installer_path="package.qwen3:install",
        )
        registry = MpsModelOperatorRegistry([spec])
        plan = types.SimpleNamespace(close=mock.Mock())
        installer = mock.Mock(return_value=plan)
        module = types.SimpleNamespace(install=installer)
        model = object()
        server_args = object()
        req_pool = object()
        kv_pool = object()
        with mock.patch(
            "importlib.import_module", return_value=module
        ) as import_module:
            result = install_mps_model_operators(
                model,
                _model_config("Qwen3ForCausalLM"),
                server_args,
                req_to_token_pool=req_pool,
                token_to_kv_pool=kv_pool,
                registry=registry,
            )
        self.assertIs(result, plan)
        import_module.assert_called_once_with("package.qwen3")
        installer.assert_called_once_with(
            model,
            mock.ANY,
            server_args,
            req_to_token_pool=req_pool,
            token_to_kv_pool=kv_pool,
        )

    def test_installer_result_must_be_closeable(self):
        spec = MpsModelOperatorSpec(
            name="qwen3",
            architectures=frozenset({"Qwen3ForCausalLM"}),
            installer_path="package.qwen3:install",
        )
        registry = MpsModelOperatorRegistry([spec])
        module = types.SimpleNamespace(install=mock.Mock(return_value=object()))
        with (
            mock.patch("importlib.import_module", return_value=module),
            self.assertRaisesRegex(TypeError, "does not implement close"),
        ):
            install_mps_model_operators(
                object(),
                _model_config("Qwen3ForCausalLM"),
                object(),
                req_to_token_pool=object(),
                token_to_kv_pool=object(),
                registry=registry,
            )


class TestModelRunnerPlatformOperatorLifecycle(unittest.TestCase):
    def _runner(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = object()
        runner.model_config = object()
        runner.server_args = object()
        runner.req_to_token_pool = object()
        runner.token_to_kv_pool = object()
        runner.platform_operator_plan = None
        return runner

    def test_bind_replaces_and_closes_previous_plan(self):
        runner = self._runner()
        previous = types.SimpleNamespace(close=mock.Mock())
        replacement = types.SimpleNamespace(close=mock.Mock())
        runner.platform_operator_plan = previous
        with mock.patch.object(
            __import__(
                "sglang.srt.model_executor.model_runner", fromlist=["current_platform"]
            ).current_platform,
            "bind_model_runtime_operators",
            return_value=replacement,
        ) as bind:
            runner._bind_platform_runtime_operators()
        previous.close.assert_called_once_with()
        self.assertIs(runner.platform_operator_plan, replacement)
        bind.assert_called_once_with(
            model=runner.model,
            model_config=runner.model_config,
            server_args=runner.server_args,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
        )

    def test_close_is_idempotent_and_clears_before_callback(self):
        runner = self._runner()

        def assert_cleared():
            self.assertIsNone(runner.platform_operator_plan)

        plan = types.SimpleNamespace(close=mock.Mock(side_effect=assert_cleared))
        runner.platform_operator_plan = plan
        runner._close_platform_runtime_operators()
        runner._close_platform_runtime_operators()
        plan.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
