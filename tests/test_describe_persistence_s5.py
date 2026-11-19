"""S5：描述事实持久性标签（trait/stage/scene）单测，全部合成文本零网络。"""
from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.cli import main as cli_main
from entitytrace.describe import (
    PERSISTENCE_LABELS,
    PERSISTENCE_PROMPT_VERSION,
    build_describe_id,
    run_describe,
)
from entitytrace.models import ModelError
from entitytrace.storage import Store
from entitytrace.util import digest

from tests.test_describe import FakeModels, FixtureCase, SERVICES


def _labels_handler(base, labels_by_short=None):
    """组合基础抽取 handler 与持久性标签 handler。"""

    def handler(stage, state, call_index):
        if stage == "describe-persistence":
            items = []
            for item in state["items"]:
                label = (labels_by_short or {}).get(item["id"], "trait")
                items.append({"id": item["id"], "label": label})
            return {"items": items}
        return base(stage, state, call_index)

    return handler


class PersistenceExportTests(FixtureCase):
    """persistence=1：打标签、落库、导出字段与计数，不删事实、不影响 stages。"""

    def _run(self, labels=None, kinds="person"):
        models = FakeModels(SERVICES, _labels_handler(self._handler, labels))
        store = Store(self.db_path)
        try:
            result = run_describe(
                store, self.run_id, services=SERVICES, models=models,
                top_k=10, kinds=kinds, concurrency=2, views=1, verify=0,
                persistence=1, out=self.out_path,
            )
        finally:
            store.close()
        return models, result

    def test_labels_export_and_counts(self):
        models, result = self._run(labels={"F1": "trait", "F2": "scene"})
        self.assertEqual(result["persistence"], 1)
        doc = result["document"]
        entity = next(e for e in doc["entities"] if e["entity_id"] == "e_shen")
        self.assertEqual(len(entity["facts"]), 2)
        self.assertEqual([f["persistence"] for f in entity["facts"]], ["trait", "scene"])
        self.assertEqual(entity["persistence_counts"],
                         {"trait": 1, "stage": 0, "scene": 1, "unlabeled": 0})
        self.assertEqual(doc["persistence"]["enabled"], True)
        self.assertEqual(doc["persistence"]["failed_entities"], 0)
        # 标签不删除事实、不影响 stages 切分（确定性零模型逻辑）。
        self.assertTrue(all(len(e["stages"]) >= 1 for e in doc["entities"] if e["facts"]))
        # 导出文件同样带字段。
        exported = json.loads(Path(self.out_path).read_text(encoding="utf-8"))
        exp_entity = next(e for e in exported["entities"] if e["entity_id"] == "e_shen")
        self.assertEqual(exp_entity["persistence_counts"]["trait"], 1)
        # 标签任务落库 describe_tasks，含 entity_id 与 labels 映射。
        store = Store(self.db_path)
        try:
            rows = store.rows(
                "SELECT id, entity_id, status, result FROM describe_tasks "
                "WHERE describe_id=? AND id LIKE 'persistence-%'",
                (result["describe_id"],),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["entity_id"], "e_shen")
            self.assertEqual(rows[0]["status"], "completed")
            payload = json.loads(rows[0]["result"])
            fact_ids = [f["id"] for f in entity["facts"]]
            self.assertEqual(set(payload["labels"]), set(fact_ids))
            self.assertIn("request_id", payload)
        finally:
            store.close()
        # 输入结构：每实体一个任务，含 entity.kind/names 与 items。
        calls = [c for c in models.calls if c["stage"] == "describe-persistence"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["temperature"], 0)
        self.assertEqual(calls[0]["state"]["entity"]["kind"], "person")
        self.assertIn("沈墨", calls[0]["state"]["entity"]["names"])
        self.assertEqual([i["id"] for i in calls[0]["state"]["items"]], ["F1", "F2"])
        self.assertIn("before", calls[0]["state"]["items"][0])
        self.assertIn("after", calls[0]["state"]["items"][0])

    def test_invalid_label_maps_to_none(self):
        _, result = self._run(labels={"F1": "unknown_tag", "F2": 7})
        entity = next(e for e in result["document"]["entities"]
                      if e["entity_id"] == "e_shen")
        self.assertEqual([f["persistence"] for f in entity["facts"]], [None, None])
        self.assertEqual(entity["persistence_counts"]["unlabeled"], 2)
        self.assertEqual(result["document"]["persistence"]["failed_entities"], 0)

    def test_invalid_output_does_not_block_export(self):
        def base(stage, state, call_index):
            text = state.get("text") or "".join(seg["text"] for seg in state.get("segments", []))
            facts = []
            if "身长八尺" in text:
                facts.append({"entity_id": "e_shen", "facet": "build", "value": "身长八尺",
                              "quote": "身长八尺", "source": "narration"})
            if "铁剑横在膝上" in text:
                facts.append({"entity_id": "e_jian", "facet": "shape", "value": "横膝铁剑",
                              "quote": "铁剑横在膝上", "source": "narration"})
            return {"facts": facts, "events": []}

        def handler(stage, state, call_index):
            if stage == "describe-persistence":
                if state["entity"]["kind"] == "person":
                    raise ModelError("invalid-output")
                return {"items": [{"id": i["id"], "label": "trait"} for i in state["items"]]}
            return base(stage, state, call_index)

        models = FakeModels(SERVICES, handler)
        store = Store(self.db_path)
        try:
            result = run_describe(
                store, self.run_id, services=SERVICES, models=models,
                top_k=10, kinds="person,object", concurrency=1, views=1, verify=0,
                persistence=1,
            )
            doc = result["document"]
            person = next(e for e in doc["entities"] if e["entity_id"] == "e_shen")
            sword = next(e for e in doc["entities"] if e["entity_id"] == "e_jian")
            self.assertEqual(doc["persistence"]["failed_entities"], 1)
            self.assertTrue(all(f["persistence"] is None for f in person["facts"]))
            self.assertEqual(person["persistence_counts"]["unlabeled"], len(person["facts"]))
            self.assertTrue(all(f["persistence"] == "trait" for f in sword["facts"]))
            rows = store.rows(
                "SELECT status, result, error FROM describe_tasks "
                "WHERE describe_id=? AND id LIKE 'persistence-%' AND entity_id='e_shen'",
                (result["describe_id"],),
            )
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("invalid-output", rows[0]["result"])
            self.assertIn("invalid-output", rows[0]["error"])
        finally:
            store.close()

    def test_resume_zero_new_persistence_calls(self):
        models1, first = self._run()
        calls_after_first = len(models1.calls)
        models2, second = self._run()
        self.assertEqual(second["describe_id"], first["describe_id"])
        self.assertEqual(len(models2.calls), 0)
        self.assertGreater(calls_after_first, 0)
        self.assertEqual(second["document"]["persistence"]["enabled"], True)
        # 全缓存复跑：导出统计的网络请求数为 0。
        self.assertEqual(second["document"]["persistence"]["requests"], 0)


class PersistenceOffTests(FixtureCase):
    """persistence=0：describe_id 与 describe-v6 逐位一致，配置不写 persistence。"""

    def test_persistence_zero_matches_v6_digest(self):
        args = (20, ["person"], ["e1"], "run", "example", "example/model")
        with patch("entitytrace.describe.DESCRIBE_VERSION", "describe-v6"):
            legacy = build_describe_id(*args)
        self.assertEqual(build_describe_id(*args, persistence=0), legacy)
        self.assertNotEqual(build_describe_id(*args, persistence=1), legacy)
        for bad in (True, 2, -1, "1", 1.0):
            with self.assertRaises(ValueError):
                build_describe_id(*args, persistence=bad)

    def test_run_persistence_zero(self):
        models = FakeModels(SERVICES, self._handler)
        store = Store(self.db_path)
        try:
            result = run_describe(
                store, self.run_id, services=SERVICES, models=models,
                top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                persistence=0,
            )
            with patch("entitytrace.describe.DESCRIBE_VERSION", "describe-v6"):
                legacy = build_describe_id(
                    10, ["person"], [e["entity_id"] for e in result["document"]["entities"]],
                    self.run_id, "example", "example/model",
                    views=1, entities_per_request=0, verify=0,
                )
            self.assertEqual(result["describe_id"], legacy)
            self.assertEqual(result["persistence"], 0)
            self.assertFalse(any(c["stage"] == "describe-persistence" for c in models.calls))
            run_row = store.rows("SELECT config FROM describe_runs WHERE id=?",
                                 (result["describe_id"],))[0]
            config = json.loads(run_row["config"])
            self.assertEqual(config["describe_version"], "describe-v6")
            self.assertNotIn("persistence", config)
            self.assertNotIn("persistence_prompt_version", config)
            entity = result["document"]["entities"][0]
            self.assertTrue(all(f["persistence"] is None for f in entity["facts"]))
            self.assertEqual(result["document"]["persistence"]["enabled"], False)
        finally:
            store.close()


class PersistenceCliTests(FixtureCase):
    """CLI：--persistence 默认 1，传 0 生效。"""

    def _run_cli(self, extra):
        class CliFake:
            def __init__(self, store, run_id, invocation, services, *args, **kwargs):
                self.services = services
                self.calls = []

            def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
                self.calls.append(stage)
                if stage == "describe-persistence":
                    value = {"items": [{"id": i["id"], "label": "trait"}
                                       for i in state["items"]]}
                else:
                    value = {"facts": [], "events": []}
                    text = state.get("text", "")
                    if "身长八尺" in text:
                        value["facts"].append({
                            "entity_id": "e_shen", "facet": "build", "value": "高大",
                            "quote": "身长八尺", "source": "narration"})
                validate(value)
                return value, "req_cli"

        out_buf, err_buf = io.StringIO(), io.StringIO()
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith("DEEPSEEK_") and not k.startswith("TYPESAFE_")
                     and not k.startswith("ENTITYTRACE_") and k != "GENERATOR_PROVIDER"}
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("entitytrace.describe.Models", CliFake), \
             contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = cli_main(["describe", self.run_id, "--db", str(self.db_path),
                           "--views", "1", "--entities-per-request", "0", "--verify", "0",
                           "--top-k", "10", "--kinds", "person", "--concurrency", "1",
                           "--config", str(self.config_path)] + extra)
        self.assertEqual(rc, 0, err_buf.getvalue())
        return json.loads(out_buf.getvalue().splitlines()[0])

    @property
    def config_path(self):
        if not hasattr(self, "_config_path"):
            self._config_path = self.root / "env.ini"
            self._config_path.write_text(
                "DEEPSEEK_API_KEY=x\n", encoding="utf-8")
        return self._config_path

    def test_default_is_one(self):
        receipt = self._run_cli([])
        self.assertEqual(receipt["persistence"], 1)

    def test_zero_disables(self):
        receipt = self._run_cli(["--persistence", "0"])
        self.assertEqual(receipt["persistence"], 0)


if __name__ == "__main__":
    unittest.main()
