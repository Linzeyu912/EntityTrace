from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.cli import main as cli_main
from entitytrace.models import Models
from tests.test_bounded_identity_run import _make_fake_http_opener
from tests.test_cli_contract import ContractFakeModels
from tests.test_corrections import ForbiddenModels

ROOT = Path(__file__).resolve().parents[1]


class HostWorkflowStageITests(unittest.TestCase):
    NOVEL_TEXT = (
        "第一章 渡口初雪\n"
        "沈墨在青石渡停船，阿墨与顾云隔岸相望，神秘旅人悄然登岸。\n"
        "第二章 客栈夜话\n"
        "沈墨回到青石渡客栈，顾云推门而入，沈墨递上热茶。\n"
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.input_path = self.root / "novel.txt"
        self.input_path.write_text(self.NOVEL_TEXT, encoding="utf-8")
        self.cfg_path = self.root / ".env"
        self.cfg_path.write_text("DEEPSEEK_API_KEY=test_secret_key\n", encoding="utf-8")
        self.db_path = self.root / "host.sqlite3"
        self.out_root = self.root / "outputs"

    def tearDown(self):
        self._tmp.cleanup()

    def _invoke(self, argv: list[str], models_cls=ForbiddenModels) -> tuple[int, dict, str]:
        out_buf, err_buf = io.StringIO(), io.StringIO()
        clean_env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("DEEPSEEK_") and not k.startswith("TYPESAFE_")
        }
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("entitytrace.pipeline.Models", models_cls), \
             contextlib.redirect_stdout(out_buf), \
             contextlib.redirect_stderr(err_buf):
            rc = cli_main(argv)
        raw_out = out_buf.getvalue()
        lines = raw_out.splitlines()
        self.assertEqual(len(lines), 1, f"stdout 必须为单行 JSON，实际：{raw_out!r}")
        payload = json.loads(lines[0])
        self.assertEqual(payload["protocol"], "entitytrace.v1")
        return rc, payload, err_buf.getvalue()

    def test_host_standard_workflow_and_stale_revision_conflict(self):
        # 1) run --json
        rc_run, run_doc, _ = self._invoke([
            "run", str(self.input_path),
            "--db", str(self.db_path),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--json",
        ], models_cls=ContractFakeModels)
        self.assertEqual(rc_run, 0)
        self.assertEqual(run_doc["status"], "completed_with_review")
        self.assertIsNone(run_doc["error"])
        run_id = run_doc["run_id"]
        rev0 = run_doc["result_revision"]
        top_review_id = run_doc["review_top"][0]["id"]

        # 2) status -> review -> entities (--surface) -> entity（全程 ForbiddenModels 保证零模型调用）
        rc_st, st_doc, _ = self._invoke([
            "status", run_id, "--db", str(self.db_path), "--output", str(self.out_root),
        ])
        self.assertEqual(rc_st, 0)
        self.assertEqual(st_doc["result_revision"], rev0)
        self.assertEqual(st_doc["status"], "completed_with_review")

        rc_rev, rev_doc, _ = self._invoke(["review", run_id, "--db", str(self.db_path)])
        self.assertEqual(rc_rev, 0)
        self.assertEqual(rev_doc["total"], 1)
        rev_item = rev_doc["items"][0]
        self.assertEqual(rev_item["id"], top_review_id)
        self.assertEqual(rev_item["status"], "pending")
        ev_span = rev_item["evidence"][0]

        rc_trav_list, trav_list_doc, _ = self._invoke([
            "entities", run_id, "--surface", "神秘旅人", "--db", str(self.db_path),
        ])
        self.assertEqual(rc_trav_list, 0)
        self.assertEqual(trav_list_doc["total"], 1)
        trav_eid = trav_list_doc["items"][0]["id"]

        rc_gu_list, gu_list_doc, _ = self._invoke([
            "entities", run_id, "--surface", "顾云", "--db", str(self.db_path),
        ])
        self.assertEqual(rc_gu_list, 0)
        gu_eid = gu_list_doc["items"][0]["id"]
        self.assertEqual(gu_list_doc["items"][0]["mention_count"], 2)

        rc_ent, ent_doc, _ = self._invoke(["entity", run_id, trav_eid, "--db", str(self.db_path)])
        self.assertEqual(rc_ent, 0)
        self.assertIn(top_review_id, ent_doc["review_ids"])
        trav_mid = ent_doc["mentions"][0]["id"]

        # 3) quote --json 取回原文上下文取证
        rc_q, q_doc, _ = self._invoke([
            "quote", run_id, str(ev_span["start"]), str(ev_span["end"]),
            "--json", "--context", "30", "--db", str(self.db_path),
        ])
        self.assertEqual(rc_q, 0)
        self.assertEqual(q_doc["result_revision"], rev0)
        self.assertEqual(q_doc["text"], "神秘旅人")
        self.assertIn("顾云", q_doc["context_before"])

        # 4) 用户确认后调用 correct assign ... --source user --expect-revision <rev0> --resolves <review_id>
        rc_cor, cor_doc, _ = self._invoke([
            "correct", run_id, "assign", trav_mid, gu_eid,
            "--source", "user",
            "--expect-revision", rev0,
            "--resolves", top_review_id,
            "--note", "用户确认神秘旅人即顾云",
            "--db", str(self.db_path),
            "--output", str(self.out_root),
        ])
        self.assertEqual(rc_cor, 0)
        self.assertIsNone(cor_doc["error"])
        self.assertEqual(cor_doc["status"], "applied")
        rev1 = cor_doc["result_revision"]
        self.assertNotEqual(rev1, rev0)

        # 5) 再次只读查询：确认待审项已标为 resolved、result_revision 更新且名册反映更正
        rc_st2, st2_doc, _ = self._invoke([
            "status", run_id, "--db", str(self.db_path), "--output", str(self.out_root),
        ])
        self.assertEqual(rc_st2, 0)
        self.assertEqual(st2_doc["result_revision"], rev1)
        self.assertEqual(st2_doc["status"], "completed")
        self.assertEqual(st2_doc["counts"]["review_items"], 0)

        rc_rev_pend, rev_pend_doc, _ = self._invoke(["review", run_id, "--db", str(self.db_path)])
        self.assertEqual(rc_rev_pend, 0)
        self.assertEqual(rev_pend_doc["total"], 0)

        rc_rev_res, rev_res_doc, _ = self._invoke([
            "review", run_id, "--status", "resolved", "--db", str(self.db_path),
        ])
        self.assertEqual(rc_rev_res, 0)
        self.assertEqual(rev_res_doc["total"], 1)
        self.assertEqual(rev_res_doc["items"][0]["id"], top_review_id)
        self.assertEqual(rev_res_doc["items"][0]["status"], "resolved")
        self.assertEqual(rev_res_doc["items"][0]["resolved_by"], cor_doc["correction_id"])

        rc_surf2, surf2_doc, _ = self._invoke([
            "entities", run_id, "--surface", "神秘旅人", "--db", str(self.db_path),
        ])
        self.assertEqual(rc_surf2, 0)
        self.assertEqual(surf2_doc["total"], 1)
        self.assertEqual(surf2_doc["items"][0]["id"], gu_eid)
        self.assertEqual(surf2_doc["items"][0]["mention_count"], 3)
        self.assertEqual(surf2_doc["items"][0]["surface_count"], 1)
        self.assertEqual(surf2_doc["items"][0]["identity_status"], "user_corrected")

        rc_gu_ent, gu_ent_doc, _ = self._invoke(["entity", run_id, gu_eid, "--db", str(self.db_path)])
        self.assertEqual(rc_gu_ent, 0)
        self.assertEqual(gu_ent_doc["result_revision"], rev1)
        self.assertEqual(gu_ent_doc["review_ids"], [])
        self.assertEqual(gu_ent_doc["link_methods"].get("user_correction"), 1)

        # 错误流程 1：仍用过期版本 rev0 提交更正 -> 返回 conflict 且退出码为 1，retryable=False
        rc_stale, stale_doc, _ = self._invoke([
            "correct", run_id, "not-entity", trav_mid,
            "--source", "user",
            "--expect-revision", rev0,
            "--db", str(self.db_path),
            "--output", str(self.out_root),
        ])
        self.assertEqual(rc_stale, 1)
        self.assertEqual(stale_doc["error"]["category"], "conflict")
        self.assertFalse(stale_doc["error"]["retryable"])

    def test_host_budget_stop_exit_2_retryable_and_same_db_resume_completes(self):
        # 错误流程 2：预算停止得到退出码 2 且 retryable=true，同库扩大预算续跑后完成（退出码 0）
        novel = self.root / "linzhou.txt"
        novel.write_text("第一章 初遇\n林舟在江边。\n第二章 重逢\n林舟回到城里。\n", encoding="utf-8")
        opener, seen_payloads = _make_fake_http_opener()

        class FakeHttpModels(Models):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, opener=opener)

        bud_db = self.root / "resume.sqlite3"
        rc1, doc1, _ = self._invoke([
            "run", str(novel),
            "--db", str(bud_db),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--max-requests", "2",
            "--json",
        ], models_cls=FakeHttpModels)
        self.assertEqual(rc1, 2)
        self.assertEqual(doc1["status"], "incomplete")
        self.assertIsNotNone(doc1["run_id"])
        self.assertEqual(doc1["error"]["category"], "budget")
        self.assertTrue(doc1["error"]["retryable"])
        self.assertEqual(len(seen_payloads), 2)

        # 宿主依据 retryable=true 同库扩大预算续跑，已完成请求零重发并顺利完成
        rc2, doc2, _ = self._invoke([
            "run", str(novel),
            "--db", str(bud_db),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--max-requests", "50",
            "--json",
        ], models_cls=FakeHttpModels)
        self.assertEqual(rc2, 0)
        self.assertEqual(doc2["run_id"], doc1["run_id"])
        self.assertIn(doc2["status"], {"completed", "completed_with_review"})
        self.assertIsNone(doc2["error"])
        self.assertTrue(doc2["stages"]["chapters"]["resumed"])
        self.assertEqual(len(seen_payloads), len(set(seen_payloads)))

    def test_plugin_readme_boundary_and_contract_consistency(self):
        plugin_doc = (ROOT / "plugin/README.md").read_text(encoding="utf-8")
        main_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("提供通用 CLI 调用说明，平台封装待核实", plugin_doc)
        self.assertIn("--source user", plugin_doc)
        self.assertIn("--source agent", plugin_doc)
        self.assertIn("error.retryable", plugin_doc)
        self.assertIn("conflict", plugin_doc)
        self.assertNotIn("entitytrace find ", plugin_doc)
        self.assertIn("plugin/README.md", main_readme)


if __name__ == "__main__":
    unittest.main()
