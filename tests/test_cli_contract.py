from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from entitytrace.cli import main as cli_main
from entitytrace.models import ModelError, Models
from entitytrace.pipeline import ALLOWED_ERROR_CATEGORIES
from entitytrace.storage import Store
from entitytrace.util import dumps
from tests.test_bounded_identity_run import _make_fake_http_opener
from tests.test_corrections import ForbiddenModels, _legacy_roster_export_bytes
from tests.test_mvp import FakeModels

ROOT = Path(__file__).resolve().parents[1]


class ContractFakeModels:
    """用于全流程 CLI 契约测试的小型假模型：产生多实体、跨章同名、非专名称呼与待审项。"""

    def __init__(self, store, run_id, invocation, services, *args, **kwargs):
        self.store = store
        self.run_id = run_id
        self.invocation = invocation
        self.peaks = {"discovery": 1}
        self.halted = {}

    def choices(self, *args, **kwargs):
        raise AssertionError("entity-roster-v1 全流程不应调用 choices")

    def generate(self, system, state, stage, validate, *args, **kwargs):
        if stage == "chapter-label":
            val = {
                "headings": [
                    {"id": c["id"], "kind": "chapter"}
                    for c in state["candidates"]
                    if c["text"].lstrip().startswith("第")
                ]
            }
        elif stage == "discovery":
            mentions = []
            for u in state["units"]:
                for surface, kind in [
                    ("沈墨", "person"),
                    ("阿墨", "person"),
                    ("顾云", "person"),
                    ("青石渡", "place"),
                    ("神秘旅人", "person"),
                ]:
                    if surface in u["text"]:
                        mentions.append({"text": surface, "kind": kind, "refs": [u["id"]]})
            val = {"mentions": mentions, "links": []}
        elif stage == "roster-c1":
            surf_map = {row["surface"]: row for row in state["surfaces"]}
            entities = []
            if "沈墨" in surf_map or "阿墨" in surf_map:
                s_list = []
                ev_ref = state["units"][0]["id"]
                for s in ("沈墨", "阿墨"):
                    if s in surf_map:
                        s_list.append({"surface": s, "surface_type": "proper_name"})
                        ev_ref = surf_map[s]["refs"][0]
                entities.append({
                    "local_id": "e_shen",
                    "name": "沈墨" if "沈墨" in surf_map else "阿墨",
                    "kind": "person",
                    "surfaces": s_list,
                    "evidence_ref": ev_ref,
                })
            if "顾云" in surf_map:
                entities.append({
                    "local_id": "e_gu",
                    "name": "顾云",
                    "kind": "person",
                    "surfaces": [{"surface": "顾云", "surface_type": "proper_name"}],
                    "evidence_ref": surf_map["顾云"]["refs"][0],
                })
            if "青石渡" in surf_map:
                entities.append({
                    "local_id": "e_place",
                    "name": "青石渡",
                    "kind": "place",
                    "surfaces": [{"surface": "青石渡", "surface_type": "proper_name"}],
                    "evidence_ref": surf_map["青石渡"]["refs"][0],
                })
            assignments = []
            if "神秘旅人" in surf_map:
                trav_row = surf_map["神秘旅人"]
                trav_mid = trav_row["mentions"][0]["mention_id"]
                assignments.append({
                    "mention_id": trav_mid,
                    "surface": "神秘旅人",
                    "ref": trav_row["refs"][0],
                    "local_id": None,
                    "reason": "章内身份未明",
                })
            val = {"entities": entities, "assignments": assignments, "not_entity": []}
        elif stage == "roster-c2-review":
            val = {
                "groups": [
                    {
                        "group_id": "g1",
                        "member_ids": [o["object_id"] for o in state["objects"]],
                        "evidence_refs": [state["objects"][0]["object_id"]],
                    }
                ],
                "uncertain_ids": [],
            }
        elif stage == "roster-c2-propose":
            val = {"proposals": []}
        elif stage == "roster-c3-link":
            val = {
                "links": [
                    {
                        "mention_id": it["mention_id"],
                        "selected_entity_id": "uncertain",
                        "reason": "无法确定归属",
                    }
                    for it in state["items"]
                ]
            }
        else:
            raise AssertionError(f"未预期阶段：{stage}")
        validate(val)
        req_id = f"fake-{stage}-{len(self.store.rows('SELECT * FROM calls')) + 1}"
        self.store.call(self.run_id, self.invocation, stage, req_id, "discovery", 1, "success", 0.01, {})
        return val, req_id


class CliContractStageFTests(unittest.TestCase):
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
        self.db_path = self.root / "contract.sqlite3"
        self.out_root = self.root / "outputs"

    def tearDown(self):
        self._tmp.cleanup()

    def _invoke(self, argv: list[str], models_cls=ForbiddenModels) -> tuple[int, str, str]:
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
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def _assert_single_json_line(self, raw_stdout: str) -> dict:
        lines = raw_stdout.splitlines()
        self.assertEqual(len(lines), 1, f"stdout 必须恰好一行 JSON，实际：{raw_stdout!r}")
        doc = json.loads(lines[0])
        self.assertEqual(doc["protocol"], "entitytrace.v1")
        return doc

    def test_run_json_single_line_no_stdout_logs_and_zero_error_export_byte_invariance(self):
        rc, out, err = self._invoke(
            [
                "run", str(self.input_path),
                "--db", str(self.db_path),
                "--output", str(self.out_root),
                "--config", str(self.cfg_path),
                "--json",
            ],
            models_cls=ContractFakeModels,
        )
        self.assertEqual(rc, 0, err)
        receipt = self._assert_single_json_line(out)
        # 进度日志全部在 stderr，stdout 无任何中文进度头
        self.assertNotIn("运行编号：", out)
        self.assertIn("运行编号：", err)
        self.assertEqual(receipt["command"], "run")
        self.assertEqual(receipt["status"], "completed_with_review")
        self.assertIsNone(receipt["error"])
        self.assertTrue(all(receipt["stages"][s]["complete"] for s in ("chapters", "discovery", "identity")))
        self.assertGreaterEqual(receipt["counts"]["entities"], 3)
        self.assertGreaterEqual(receipt["counts"]["review_items"], 1)
        self.assertLessEqual(len(receipt["review_top"]), 10)
        self.assertEqual(set(receipt["review_top"][0].keys()), {"id", "stage", "reason", "impact"})

        # 零错误运行导出逐字节等于阶段 G/F 前基准快照，且 manifest.state.last_error 仍为 None
        run_id = receipt["run_id"]
        out_dir = Path(receipt["output_dir"])
        store = Store(self.db_path)
        try:
            legacy_res, legacy_rev, legacy_man = _legacy_roster_export_bytes(store, run_id)
        finally:
            store.close()
        self.assertEqual((out_dir / "result.json").read_bytes(), legacy_res)
        self.assertEqual((out_dir / "review.json").read_bytes(), legacy_rev)
        self.assertEqual((out_dir / "manifest.json").read_bytes(), legacy_man)
        man_doc = json.loads(legacy_man)
        self.assertIsNone(man_doc["state"]["last_error"])

        # 不加 --json 时 run 的 stdout 文本保持原样
        plain_db = self.root / "plain.sqlite3"
        plain_out_dir = self.root / "plain_out"
        rc_plain, out_plain, err_plain = self._invoke(
            [
                "run", str(self.input_path),
                "--db", str(plain_db),
                "--output", str(plain_out_dir),
                "--config", str(self.cfg_path),
            ],
            models_cls=ContractFakeModels,
        )
        self.assertEqual(rc_plain, 0)
        self.assertEqual(err_plain, "")
        self.assertIn(f"运行编号：{run_id}", out_plain)
        self.assertIn("输出目录：", out_plain)

    def test_queries_pagination_surface_snippet_boundaries_quote_and_zero_calls(self):
        rc, out, _ = self._invoke(
            [
                "run", str(self.input_path),
                "--db", str(self.db_path),
                "--output", str(self.out_root),
                "--config", str(self.cfg_path),
                "--json",
            ],
            models_cls=ContractFakeModels,
        )
        self.assertEqual(rc, 0)
        run_receipt = self._assert_single_json_line(out)
        run_id = run_receipt["run_id"]
        rev0 = run_receipt["result_revision"]

        store = Store(self.db_path)
        try:
            calls_before = len(store.rows("SELECT * FROM calls"))
        finally:
            store.close()
        self.assertGreater(calls_before, 0)

        # 删除磁盘导出目录，验证全部查询均从 SQLite 内存实时构建，不读也不重写磁盘导出文件
        shutil.rmtree(self.out_root)
        self.assertFalse(self.out_root.exists())

        # 1) status 查询
        rc_st, out_st, _ = self._invoke(["status", run_id, "--db", str(self.db_path), "--output", str(self.out_root)])
        self.assertEqual(rc_st, 0)
        st_doc = self._assert_single_json_line(out_st)
        self.assertEqual(st_doc["command"], "status")
        self.assertEqual(st_doc["result_revision"], rev0)
        self.assertEqual(st_doc["status"], "completed_with_review")
        self.assertIsNone(st_doc["last_error"])
        self.assertIsNone(st_doc["error"])

        # 2) entities 分页（total / next_offset）与筛选（--kind / --min-mentions / --surface）及上限拒绝
        rc_p1, out_p1, _ = self._invoke(["entities", run_id, "--limit", "2", "--offset", "0", "--db", str(self.db_path)])
        self.assertEqual(rc_p1, 0)
        p1 = self._assert_single_json_line(out_p1)
        self.assertEqual(p1["result_revision"], rev0)
        self.assertGreaterEqual(p1["total"], 4)
        self.assertEqual(len(p1["items"]), 2)
        self.assertEqual(p1["next_offset"], 2)
        self.assertEqual(
            set(p1["items"][0].keys()),
            {"id", "name", "kind", "mention_count", "first_chapter_id", "identity_status"},
        )
        # 按 mention_count 降序、id 升序
        self.assertGreaterEqual(p1["items"][0]["mention_count"], p1["items"][1]["mention_count"])

        rc_last, out_last, _ = self._invoke(
            ["entities", run_id, "--limit", "50", "--offset", "2", "--db", str(self.db_path)]
        )
        self.assertEqual(rc_last, 0)
        p_last = self._assert_single_json_line(out_last)
        self.assertIsNone(p_last["next_offset"])

        # --limit > 200 按 input 拒绝，退出码 1
        rc_over, out_over, _ = self._invoke(["entities", run_id, "--limit", "201", "--db", str(self.db_path)])
        self.assertEqual(rc_over, 1)
        over_doc = self._assert_single_json_line(out_over)
        self.assertEqual(over_doc["error"]["category"], "input")
        self.assertFalse(over_doc["error"]["retryable"])

        # --surface 精确匹配并附带 surface_count（沈墨实体含 3 处“沈墨”和 1 处“阿墨”，总提及 4，查“阿墨”时 surface_count=1）
        rc_surf, out_surf, _ = self._invoke(["entities", run_id, "--surface", "阿墨", "--db", str(self.db_path)])
        self.assertEqual(rc_surf, 0)
        surf_doc = self._assert_single_json_line(out_surf)
        self.assertEqual(surf_doc["total"], 1)
        shen_item = surf_doc["items"][0]
        self.assertEqual(shen_item["name"], "沈墨")
        self.assertEqual(shen_item["mention_count"], 4)
        self.assertEqual(shen_item["surface_count"], 1)

        # find 命令已裁撤，不得存在
        rc_find, _, _ = self._invoke(["find", run_id, "阿墨", "--db", str(self.db_path)])
        self.assertEqual(rc_find, 1)

        # 3) entity 详情与 snippet 绝不跨章节边界（使用最大 --context 400 验证）
        shen_id = shen_item["id"]
        rc_ent, out_ent, _ = self._invoke(
            ["entity", run_id, shen_id, "--context", "400", "--limit", "2", "--db", str(self.db_path)]
        )
        self.assertEqual(rc_ent, 0)
        ent_doc = self._assert_single_json_line(out_ent)
        self.assertEqual(ent_doc["entity_id"], shen_id)
        self.assertEqual(ent_doc["aliases"], ["沈墨", "阿墨"])
        self.assertIn("chapter_local", ent_doc["link_methods"])
        self.assertEqual(ent_doc["total"], 4)
        self.assertEqual(ent_doc["next_offset"], 2)
        for m in ent_doc["mentions"]:
            self.assertEqual(set(m.keys()), {"id", "chapter_id", "start", "end", "text", "link_method", "snippet"})
            # 第一章提及的 snippet 绝不能包含“第二章”，第二章提及的 snippet 绝不能包含“第一章”
            if m["start"] < self.NOVEL_TEXT.index("第二章"):
                self.assertNotIn("第二章", m["snippet"])
            else:
                self.assertNotIn("第一章", m["snippet"])

        # --context > 400 或未知实体 ID 返回 input 错误（退出码 1）
        rc_ctx_bad, out_ctx_bad, _ = self._invoke(
            ["entity", run_id, shen_id, "--context", "401", "--db", str(self.db_path)]
        )
        self.assertEqual(rc_ctx_bad, 1)
        self.assertEqual(self._assert_single_json_line(out_ctx_bad)["error"]["category"], "input")

        rc_ent_bad, out_ent_bad, _ = self._invoke(
            ["entity", run_id, "unknown_entity_id", "--db", str(self.db_path)]
        )
        self.assertEqual(rc_ent_bad, 1)
        self.assertEqual(self._assert_single_json_line(out_ent_bad)["error"]["category"], "input")

        # 4) review 查询（默认 pending，按 impact 降序、id 升序）
        rc_rev, out_rev, _ = self._invoke(["review", run_id, "--db", str(self.db_path)])
        self.assertEqual(rc_rev, 0)
        rev_doc = self._assert_single_json_line(out_rev)
        self.assertEqual(rev_doc["result_revision"], rev0)
        self.assertGreaterEqual(rev_doc["total"], 1)
        self.assertTrue(all(it["status"] == "pending" for it in rev_doc["items"]))
        self.assertTrue(all(isinstance(it["impact"], int) for it in rev_doc["items"]))

        # 关联神秘旅人暂定实体的 review_ids 非空
        rc_trav_list, out_trav_list, _ = self._invoke(
            ["entities", run_id, "--surface", "神秘旅人", "--db", str(self.db_path)]
        )
        self.assertEqual(rc_trav_list, 0)
        trav_id = self._assert_single_json_line(out_trav_list)["items"][0]["id"]
        rc_trav_ent, out_trav_ent, _ = self._invoke(["entity", run_id, trav_id, "--db", str(self.db_path)])
        self.assertEqual(rc_trav_ent, 0)
        self.assertGreaterEqual(len(self._assert_single_json_line(out_trav_ent)["review_ids"]), 1)

        # 5) quote：--json --context N vs 纯文本输出保持原样
        first_m = ent_doc["mentions"][0]
        rc_qj, out_qj, _ = self._invoke([
            "quote", run_id, str(first_m["start"]), str(first_m["end"]),
            "--json", "--context", "6", "--db", str(self.db_path),
        ])
        self.assertEqual(rc_qj, 0)
        qj_doc = self._assert_single_json_line(out_qj)
        self.assertEqual(qj_doc["result_revision"], rev0)
        self.assertEqual(qj_doc["text"], first_m["text"])
        self.assertEqual(qj_doc["chapter_id"], first_m["chapter_id"])
        self.assertLessEqual(len(qj_doc["context_before"]), 6)
        self.assertLessEqual(len(qj_doc["context_after"]), 6)

        rc_qp, out_qp, _ = self._invoke([
            "quote", run_id, str(first_m["start"]), str(first_m["end"]), "--db", str(self.db_path),
        ])
        self.assertEqual(rc_qp, 0)
        self.assertEqual(out_qp, first_m["text"] + "\n")

        # 磁盘导出目录仍不存在（查询不产生副作用），且 calls 行数零变化
        self.assertFalse(self.out_root.exists())
        store = Store(self.db_path)
        try:
            self.assertEqual(len(store.rows("SELECT * FROM calls")), calls_before)
        finally:
            store.close()

    def test_all_eight_error_categories_and_exit_codes(self):
        seen_categories: set[str] = set()

        # 1) input（运行未建立 -> status="failed", run_id=None, exit=1, retryable=False）
        rc_in, out_in, _ = self._invoke([
            "run", str(self.root / "missing.txt"),
            "--db", str(self.root / "in.sqlite3"),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--json",
        ])
        self.assertEqual(rc_in, 1)
        doc_in = self._assert_single_json_line(out_in)
        self.assertEqual(doc_in["status"], "failed")
        self.assertIsNone(doc_in["run_id"])
        self.assertEqual(doc_in["error"]["category"], "input")
        self.assertFalse(doc_in["error"]["retryable"])
        seen_categories.add(doc_in["error"]["category"])

        # 解码失败同样归为 input（exit=1）
        bad_bytes = self.root / "bad_utf8.txt"
        bad_bytes.write_bytes(b"\xff\xfe\xfd")
        rc_enc, out_enc, _ = self._invoke([
            "run", str(bad_bytes),
            "--db", str(self.root / "enc.sqlite3"),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--json",
        ])
        self.assertEqual(rc_enc, 1)
        self.assertEqual(self._assert_single_json_line(out_enc)["error"]["category"], "input")

        # 2) config（缺少密钥 -> status="failed", run_id=None, exit=1, retryable=False）
        empty_cfg = self.root / "empty.env"
        empty_cfg.write_text("", encoding="utf-8")
        rc_cfg, out_cfg, _ = self._invoke([
            "run", str(self.input_path),
            "--db", str(self.root / "cfg.sqlite3"),
            "--output", str(self.out_root),
            "--config", str(empty_cfg),
            "--json",
        ])
        self.assertEqual(rc_cfg, 1)
        doc_cfg = self._assert_single_json_line(out_cfg)
        self.assertEqual(doc_cfg["status"], "failed")
        self.assertIsNone(doc_cfg["run_id"])
        self.assertEqual(doc_cfg["error"]["category"], "config")
        self.assertFalse(doc_cfg["error"]["retryable"])
        seen_categories.add(doc_cfg["error"]["category"])

        # 3) budget（保留断点 -> status="incomplete", exit=2, retryable=True；且未完成运行支持 status 查询）
        opener_ok, _ = _make_fake_http_opener()

        class BudgetHttpModels(Models):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, opener=opener_ok)

        bud_db = self.root / "bud.sqlite3"
        rc_bud, out_bud, _ = self._invoke([
            "run", str(self.input_path),
            "--db", str(bud_db),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--max-requests", "1",
            "--json",
        ], models_cls=BudgetHttpModels)
        self.assertEqual(rc_bud, 2)
        doc_bud = self._assert_single_json_line(out_bud)
        self.assertEqual(doc_bud["status"], "incomplete")
        self.assertIsNotNone(doc_bud["run_id"])
        self.assertEqual(doc_bud["error"]["category"], "budget")
        self.assertTrue(doc_bud["error"]["retryable"])
        seen_categories.add(doc_bud["error"]["category"])

        # 未完成运行可查 status
        rc_st_inc, out_st_inc, _ = self._invoke([
            "status", doc_bud["run_id"], "--db", str(bud_db), "--output", str(self.out_root),
        ])
        self.assertEqual(rc_st_inc, 0)
        st_inc = self._assert_single_json_line(out_st_inc)
        self.assertEqual(st_inc["status"], "incomplete")
        self.assertEqual(st_inc["last_error"]["category"], "budget")

        # 4) network（断网中断 -> status="incomplete", exit=2, retryable=True）
        opener_net, _ = _make_fake_http_opener(fail_network=True)

        class NetworkFailModels(Models):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, attempts=1, opener=opener_net)

        with patch("entitytrace.models.time.sleep"):
            rc_net, out_net, _ = self._invoke([
                "run", str(self.input_path),
                "--db", str(self.root / "net.sqlite3"),
                "--output", str(self.out_root),
                "--config", str(self.cfg_path),
                "--json",
            ], models_cls=NetworkFailModels)
        self.assertEqual(rc_net, 2)
        doc_net = self._assert_single_json_line(out_net)
        self.assertEqual(doc_net["status"], "incomplete")
        self.assertEqual(doc_net["error"]["category"], "network")
        self.assertTrue(doc_net["error"]["retryable"])
        seen_categories.add(doc_net["error"]["category"])

        # 5) quota（硬额度耗尽 -> 数据库保留断点 exit=2，但 retryable=False，宿主不得自动重跑）
        def quota_opener(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            state = json.loads(payload["messages"][1]["content"])
            if "candidates" in state:
                return opener_ok(request, timeout=timeout)
            raise urllib.error.HTTPError(
                request.full_url,
                402,
                "Payment Required",
                {},
                io.BytesIO(b'{"detail":{"error_type":"billing_error","message":"insufficient balance"}}'),
            )

        class QuotaFailModels(Models):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, attempts=1, opener=quota_opener)

        rc_q, out_q, _ = self._invoke([
            "run", str(self.input_path),
            "--db", str(self.root / "quota.sqlite3"),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--json",
        ], models_cls=QuotaFailModels)
        self.assertEqual(rc_q, 2)
        doc_q = self._assert_single_json_line(out_q)
        self.assertEqual(doc_q["status"], "incomplete")
        self.assertEqual(doc_q["error"]["category"], "quota")
        self.assertFalse(doc_q["error"]["retryable"])
        seen_categories.add(doc_q["error"]["category"])

        # 6) model_output（发现任务连续格式错误未抛异常 -> 回执 error.category="model_output", retryable=True, exit=2，且不写回 manifest.json）
        class FormatFailDiscoveryModels(ContractFakeModels):
            def generate(self, system, state, stage, validate, *args, **kwargs):
                if stage == "discovery":
                    raise ModelError("discovery 返回格式不符合约定")
                return super().generate(system, state, stage, validate, *args, **kwargs)

        fmt_out = self.root / "fmt_out"
        rc_fmt, out_fmt, _ = self._invoke([
            "run", str(self.input_path),
            "--db", str(self.root / "fmt.sqlite3"),
            "--output", str(fmt_out),
            "--config", str(self.cfg_path),
            "--json",
        ], models_cls=FormatFailDiscoveryModels)
        self.assertEqual(rc_fmt, 2)
        doc_fmt = self._assert_single_json_line(out_fmt)
        self.assertEqual(doc_fmt["status"], "incomplete")
        self.assertEqual(doc_fmt["error"]["category"], "model_output")
        self.assertTrue(doc_fmt["error"]["retryable"])
        self.assertIn("2 个任务", doc_fmt["error"]["message"])
        fmt_manifest = json.loads(Path(doc_fmt["files"]["manifest"]).read_text(encoding="utf-8"))
        self.assertIsNone(fmt_manifest["state"]["last_error"])
        seen_categories.add(doc_fmt["error"]["category"])

        # 7) conflict（过期 result_revision -> exit=1, retryable=False）
        ok_db = self.root / "ok.sqlite3"
        rc_ok, out_ok, _ = self._invoke([
            "run", str(self.input_path),
            "--db", str(ok_db),
            "--output", str(self.out_root),
            "--config", str(self.cfg_path),
            "--json",
        ], models_cls=ContractFakeModels)
        self.assertEqual(rc_ok, 0)
        ok_doc = self._assert_single_json_line(out_ok)
        ents_doc = self._assert_single_json_line(
            self._invoke(["entities", ok_doc["run_id"], "--db", str(ok_db)])[1]
        )
        e0, e1 = ents_doc["items"][0]["id"], ents_doc["items"][1]["id"]
        rc_conf, out_conf, _ = self._invoke([
            "correct", ok_doc["run_id"], "merge", e0, e1,
            "--source", "user", "--expect-revision", "outdated_revision",
            "--db", str(ok_db), "--output", str(self.out_root),
        ])
        self.assertEqual(rc_conf, 1)
        doc_conf = self._assert_single_json_line(out_conf)
        self.assertEqual(doc_conf["error"]["category"], "conflict")
        self.assertFalse(doc_conf["error"]["retryable"])
        seen_categories.add(doc_conf["error"]["category"])

        # 8) local（原文指纹被篡改 -> exit=1, retryable=False）
        store = Store(ok_db)
        try:
            store.execute("UPDATE sources SET text = text || '篡改'")
        finally:
            store.close()
        rc_loc, out_loc, _ = self._invoke(["status", ok_doc["run_id"], "--db", str(ok_db)])
        self.assertEqual(rc_loc, 1)
        doc_loc = self._assert_single_json_line(out_loc)
        self.assertEqual(doc_loc["error"]["category"], "local")
        self.assertFalse(doc_loc["error"]["retryable"])
        seen_categories.add(doc_loc["error"]["category"])

        # 8 个允许类别全部实测覆盖且无任何越界类别
        self.assertEqual(seen_categories, set(ALLOWED_ERROR_CATEGORIES))

    def test_groups_v2_run_rejected_by_readonly_queries_with_input_category(self):
        v2_cfg = self.root / "v2.env"
        v2_cfg.write_text("DEEPSEEK_API_KEY=test\nTYPESAFE_API_KEY=test\n", encoding="utf-8")
        v2_db = self.root / "v2.sqlite3"
        FakeModels.fail_once = False
        rc_run, out_run, _ = self._invoke([
            "run", str(self.input_path),
            "--db", str(v2_db),
            "--output", str(self.out_root),
            "--config", str(v2_cfg),
            "--identity-algorithm", "groups-v2",
            "--json",
        ], models_cls=FakeModels)
        self.assertEqual(rc_run, 0)
        v2_id = self._assert_single_json_line(out_run)["run_id"]

        for argv in [
            ["status", v2_id, "--db", str(v2_db)],
            ["entities", v2_id, "--db", str(v2_db)],
            ["entity", v2_id, "any_id", "--db", str(v2_db)],
            ["review", v2_id, "--db", str(v2_db)],
            ["quote", v2_id, "0", "3", "--json", "--db", str(v2_db)],
        ]:
            with self.subTest(cmd=argv[0]):
                rc_q, out_q, _ = self._invoke(argv)
                self.assertEqual(rc_q, 1)
                doc_q = self._assert_single_json_line(out_q)
                self.assertEqual(doc_q["error"]["category"], "input")
                self.assertFalse(doc_q["error"]["retryable"])

    def test_query_module_isolation_no_models_no_tools_no_book_names(self):
        src = (ROOT / "entitytrace/query.py").read_text(encoding="utf-8")
        for forbidden in ["entity-gold", "evaluate_identity_gold", "Pride", "九命奇冤", "伤逝", "Second Stain", "Elizabeth", "Darcy"]:
            self.assertNotIn(forbidden, src)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith("tools"))
                self.assertNotIn("models", node.module)


if __name__ == "__main__":
    unittest.main()
