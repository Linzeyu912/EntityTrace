"""生成服务商可配置（deepseek | openai）离线测试。"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from entitytrace.cli import main as cli_main
from entitytrace.config import (
    OPENAI_DEFAULT_CONCURRENCY,
    OPENAI_DEFAULT_TIMEOUT_SECONDS,
    Service,
    load_services,
)
from entitytrace.models import ModelError, Models
from entitytrace.pipeline import classify_exception, run_pipeline
from entitytrace.storage import Store
from entitytrace.util import digest, dumps
from tests.test_mvp import FakeModels

_ENV_PREFIXES = ("DEEPSEEK_", "TYPESAFE_", "ENTITYTRACE_OPENAI_", "GENERATOR_")


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith(_ENV_PREFIXES)}


def _write_text(root: Path, text: str = "第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n") -> Path:
    path = root / "input.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _write_env(root: Path, body: str) -> Path:
    cfg = root / ".env"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def _recording_models_factory(record: dict, models_cls=FakeModels):
    class RecordingModels(models_cls):
        def __init__(self, *args, **kwargs):
            record["args"] = args
            record["kwargs"] = dict(kwargs)
            super().__init__(*args, **kwargs)

    return RecordingModels


class ProviderConfigTests(unittest.TestCase):
    """生成服务商解析、参数校验与分类测试。"""

    def test_default_provider_is_deepseek(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _write_env(root, "DEEPSEEK_API_KEY=ds_test\n")
            with patch.dict(os.environ, _clean_env(), clear=True):
                svc = load_services(cfg)["discovery"]
                self.assertEqual(svc.provider, "deepseek")
                self.assertEqual(svc.url, "https://api.deepseek.com")
                self.assertEqual(svc.model, "deepseek-flash")
                self.assertEqual(svc.key, "ds_test")
                self.assertEqual(set(svc.public()), {"service", "url", "model"})

    def test_openai_parses_url_model_key_extra_body_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _write_env(
                Path(d),
                "GENERATOR_PROVIDER=openai\n"
                "ENTITYTRACE_OPENAI_BASE_URL=https://api.example.com/v1/\n"
                "ENTITYTRACE_OPENAI_MODEL=example/model\n"
                "ENTITYTRACE_OPENAI_API_KEY=test_key\n"
                'ENTITYTRACE_OPENAI_EXTRA_BODY={"reasoning_effort":"none"}\n'
                "ENTITYTRACE_OPENAI_TIMEOUT_SECONDS=300\n",
            )
            with patch.dict(os.environ, _clean_env(), clear=True):
                svc = load_services(cfg)["discovery"]
                self.assertEqual(svc.provider, "openai")
                self.assertEqual(svc.url, "https://api.example.com/v1")
                self.assertEqual(svc.model, "example/model")
                self.assertEqual(svc.key, "test_key")
                self.assertEqual(svc.extra_body, {"reasoning_effort": "none"})
                self.assertEqual(svc.timeout_seconds, 300.0)
                self.assertEqual(set(svc.public()), {"service", "url", "model"})

    def test_openai_missing_required_env_vars_classified_as_config_error(self):
        cases = [
            "GENERATOR_PROVIDER=openai\nENTITYTRACE_OPENAI_MODEL=example/model\nENTITYTRACE_OPENAI_API_KEY=k\n",
            "GENERATOR_PROVIDER=openai\nENTITYTRACE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1\nENTITYTRACE_OPENAI_API_KEY=k\n",
            "GENERATOR_PROVIDER=openai\nENTITYTRACE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1\nENTITYTRACE_OPENAI_MODEL=example/model\n",
            "GENERATOR_PROVIDER=openai\nENTITYTRACE_OPENAI_BASE_URL=ftp://127.0.0.1/v1\nENTITYTRACE_OPENAI_MODEL=example/model\nENTITYTRACE_OPENAI_API_KEY=k\n",
            "GENERATOR_PROVIDER=openai\nENTITYTRACE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1\nENTITYTRACE_OPENAI_MODEL=example/model\nENTITYTRACE_OPENAI_API_KEY=k\nENTITYTRACE_OPENAI_EXTRA_BODY=not-json\n",
            "GENERATOR_PROVIDER=openai\nENTITYTRACE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1\nENTITYTRACE_OPENAI_MODEL=example/model\nENTITYTRACE_OPENAI_API_KEY=k\nENTITYTRACE_OPENAI_EXTRA_BODY=[1,2]\n",
            "GENERATOR_PROVIDER=openai\nENTITYTRACE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1\nENTITYTRACE_OPENAI_MODEL=example/model\nENTITYTRACE_OPENAI_API_KEY=k\nENTITYTRACE_OPENAI_TIMEOUT_SECONDS=0\n",
        ]
        for body in cases:
            with tempfile.TemporaryDirectory() as d:
                cfg = _write_env(Path(d), body)
                with patch.dict(os.environ, _clean_env(), clear=True):
                    with self.assertRaises(ValueError) as ctx:
                        load_services(cfg)
                err = classify_exception(ctx.exception)
                self.assertEqual(err["category"], "config", body)
                self.assertFalse(err["retryable"])

    def test_illegal_provider_value_is_config_error(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _write_env(Path(d), "DEEPSEEK_API_KEY=ds_test\nGENERATOR_PROVIDER=bogus\n")
            with patch.dict(os.environ, _clean_env(), clear=True):
                with self.assertRaises(ValueError) as ctx:
                    load_services(cfg)
            self.assertEqual(ctx.exception.et_category, "config")
            self.assertEqual(classify_exception(ctx.exception)["category"], "config")

    def test_deepseek_url_whitelist_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _write_env(
                Path(d),
                "DEEPSEEK_API_KEY=ds_test\nGENERATOR_PROVIDER=deepseek\nDEEPSEEK_BASE_URL=https://api.evil.example\n",
            )
            with patch.dict(os.environ, _clean_env(), clear=True):
                with self.assertRaises(ValueError):
                    load_services(cfg)


class OpenAIPayloadAndFenceTests(unittest.TestCase):
    """通用兼容路径的 extra_body 合并/删除、request_id 计算与代码围栏剥离。"""

    def _models(self, store: Store, provider: str, opener, extra_body: dict | None = None):
        if provider == "openai":
            svc = Service(
                "discovery",
                "http://127.0.0.1:8000/v1",
                "example/model",
                "test_key",
                provider="openai",
                extra_body=extra_body or {},
            )
        else:
            svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")
        return Models(store, "run-openai", "inv-1", {"discovery": svc}, attempts=1, opener=opener)

    def test_extra_body_merge_and_null_deletion_before_request_id(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "db.sqlite3")
            try:
                sent: list[dict] = []

                def opener(request, **kwargs):
                    sent.append(json.loads(request.data.decode("utf-8")))
                    body = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                models = self._models(
                    store,
                    "openai",
                    opener,
                    extra_body={"reasoning": {"effort": "medium"}, "response_format": None},
                )
                _, req_id = models.generate("sys", {"x": 1}, "discovery", lambda v: None)
                self.assertEqual(len(sent), 1)
                payload = sent[0]
                self.assertNotIn("thinking", payload)
                self.assertNotIn("response_format", payload)
                self.assertEqual(payload.get("reasoning"), {"effort": "medium"})
                self.assertEqual(payload["max_tokens"], 8000)
                self.assertEqual(payload["model"], "example/model")
                expected_id = digest(["http-v1", models.services["discovery"].public(), "/chat/completions", payload])
                self.assertEqual(req_id, expected_id)
            finally:
                store.close()

    def test_openai_strips_markdown_fence_including_cached(self):
        fenced = '```json\n{"a": 1}\n```'
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "db.sqlite3")
            try:
                calls: list[int] = []

                def opener(request, **kwargs):
                    calls.append(1)
                    body = {"choices": [{"finish_reason": "stop", "message": {"content": fenced}}]}
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                seen: list[dict] = []

                def validate(value):
                    seen.append(value)

                models = self._models(store, "openai", opener)
                value, request_id = models.generate("sys", {"x": 1}, "discovery", validate)
                self.assertEqual(value, {"a": 1})
                self.assertEqual(seen, [{"a": 1}])
                self.assertEqual(len(calls), 1)
                value2, request_id2 = models.generate("sys", {"x": 1}, "discovery", validate)
                self.assertEqual((value2, request_id2), ({"a": 1}, request_id))
                self.assertEqual(len(calls), 1)
            finally:
                store.close()

    def test_deepseek_does_not_strip_fence(self):
        fenced = '```json\n{"a": 1}\n```'
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "db.sqlite3")
            try:
                def opener(request, **kwargs):
                    body = {"choices": [{"finish_reason": "stop", "message": {"content": fenced}}]}
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                models = self._models(store, "deepseek", opener)
                with self.assertRaises(ModelError):
                    models.generate("sys", {"x": 1}, "discovery", lambda v: None)
            finally:
                store.close()


class DeepSeekInvariantTests(unittest.TestCase):
    """deepseek 路径负载、request_id、run_id 配置摘要、超时 120、并发 6 全部不变。"""

    def test_default_deepseek_payload_request_id_and_public(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "db.sqlite3")
            try:
                sent: list[dict] = []

                def opener(request, **kwargs):
                    sent.append(json.loads(request.data.decode("utf-8")))
                    body = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
                    return io.BytesIO(json.dumps(body).encode("utf-8"))

                svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")
                self.assertEqual(svc.provider, "deepseek")
                self.assertEqual(svc.public(), {"service": "discovery", "url": svc.url, "model": "deepseek-flash"})
                models = Models(store, "r", "i", {"discovery": svc}, opener=opener)
                sys_prompt, user_state = "测试提示词", {"units": [{"id": "u0_5", "text": "测试"}]}
                expected_payload = {
                    "model": "deepseek-flash",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": dumps(user_state)},
                    ],
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                    "max_tokens": 8000,
                }
                _, request_id = models.generate(sys_prompt, user_state, "discovery", lambda v: None)
                self.assertEqual(dumps(sent[0]), dumps(expected_payload))
                self.assertEqual(request_id, digest(["http-v1", svc.public(), "/chat/completions", expected_payload]))
            finally:
                store.close()

    def test_default_run_deepseek_timeout120_concurrency6_and_stable_run_id(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = _write_text(root)
            cfg = _write_env(root, "DEEPSEEK_API_KEY=ds_test\n")
            record: dict = {}
            recording = _recording_models_factory(record)
            with patch.dict(os.environ, _clean_env(), clear=True), patch(
                "entitytrace.pipeline.Models", recording
            ):
                run_id, output, complete = run_pipeline(
                    text, root / "db.sqlite3", root / "out", cfg, log=lambda _: None
                )
                run_id2, _, complete2 = run_pipeline(
                    text, root / "db.sqlite3", root / "out", cfg, log=lambda _: None
                )
            self.assertTrue(complete)
            self.assertTrue(complete2)
            self.assertEqual(run_id, run_id2)
            kwargs = record["kwargs"]
            self.assertEqual(kwargs.get("timeout"), 120)
            self.assertEqual(record["args"][4], 6)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            config = manifest["config"]
            self.assertEqual(
                set(config),
                {
                    "implementation",
                    "models",
                    "discovery_concurrency",
                    "decision_concurrency",
                    "task_chars",
                    "offset_unit",
                    "scope",
                    "identity_algorithm",
                },
            )
            self.assertEqual(config["discovery_concurrency"], 6)
            self.assertEqual(
                config["models"]["discovery"],
                {"service": "discovery", "url": "https://api.deepseek.com", "model": "deepseek-flash"},
            )


class OpenAITimeoutConcurrencyTests(unittest.TestCase):
    """openai 默认超时 420、CLI 默认并发 8（deepseek 为 6）；显式并发与超时优先。"""

    def _run_cli_with_openai(self, root: Path, extra_args: list[str] | None = None, extra_env: str = ""):
        text = _write_text(root)
        cfg = _write_env(
            root,
            "GENERATOR_PROVIDER=openai\n"
            "ENTITYTRACE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1\n"
            "ENTITYTRACE_OPENAI_MODEL=example/model\n"
            "ENTITYTRACE_OPENAI_API_KEY=test_key\n"
            + extra_env,
        )
        record: dict = {}
        recording = _recording_models_factory(record)
        out_buf, err_buf = io.StringIO(), io.StringIO()
        argv = [
            "run", str(text),
            "--db", str(root / "db.sqlite3"),
            "--output", str(root / "out"),
            "--config", str(cfg),
            "--json",
            *(extra_args or []),
        ]
        with patch.dict(os.environ, _clean_env(), clear=True), patch(
            "entitytrace.pipeline.Models", recording
        ), redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = cli_main(argv)
        self.assertEqual(rc, 0, err_buf.getvalue())
        receipt = json.loads(out_buf.getvalue().splitlines()[0])
        return record, receipt, Path(receipt["output_dir"])

    def test_openai_default_timeout_and_concurrency(self):
        with tempfile.TemporaryDirectory() as d:
            record, receipt, output = self._run_cli_with_openai(Path(d))
            kwargs = record["kwargs"]
            self.assertEqual(kwargs.get("timeout"), OPENAI_DEFAULT_TIMEOUT_SECONDS)
            self.assertEqual(record["args"][4], OPENAI_DEFAULT_CONCURRENCY)
            self.assertEqual(receipt["discovery_concurrency"], OPENAI_DEFAULT_CONCURRENCY)
            self.assertEqual(receipt["generator_provider"], "openai")
            self.assertEqual(receipt["generator_model"], "example/model")
            self.assertEqual(receipt["generator_timeout_seconds"], 420)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            latest = next(i for i in manifest["invocations"] if i["id"] == manifest["state"]["last_invocation"])
            self.assertEqual(
                latest["generator"],
                {
                    "generator_provider": "openai",
                    "generator_model": "example/model",
                    "generator_timeout_seconds": 420,
                    "discovery_concurrency": OPENAI_DEFAULT_CONCURRENCY,
                },
            )

    def test_explicit_discovery_concurrency_and_timeout_win(self):
        with tempfile.TemporaryDirectory() as d:
            record, receipt, _ = self._run_cli_with_openai(
                Path(d),
                ["--discovery-concurrency", "5"],
                extra_env="ENTITYTRACE_OPENAI_TIMEOUT_SECONDS=180\n",
            )
            self.assertEqual(record["args"][4], 5)
            self.assertEqual(record["kwargs"].get("timeout"), 180)
            self.assertEqual(receipt["discovery_concurrency"], 5)
            self.assertEqual(receipt["generator_timeout_seconds"], 180)


class AuthErrorMappingTests(unittest.TestCase):
    """HTTP 401 归入 config。"""

    def _401_opener(self, request, **kwargs):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {},
            io.BytesIO(b'{"detail":{"error_type":"unauthorized"}}'),
        )

    def test_401_maps_to_config_for_both_providers(self):
        for provider, svc in (
            ("deepseek", Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")),
            ("openai", Service("discovery", "http://127.0.0.1:8000/v1", "example/model", "test_key", provider="openai")),
        ):
            with tempfile.TemporaryDirectory() as d:
                store = Store(Path(d) / "db.sqlite3")
                try:
                    models = Models(store, "r", "i", {"discovery": svc}, attempts=1, opener=self._401_opener)
                    with self.assertRaises(ModelError) as ctx:
                        models.generate("sys", {"x": 1}, "discovery", lambda v: None)
                    err = classify_exception(ctx.exception)
                    self.assertEqual(err["category"], "config", provider)
                    self.assertFalse(err["retryable"])
                    self.assertEqual(ctx.exception.status, 401)
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()
