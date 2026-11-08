from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.cli import main as cli_main
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.pipeline import export_run
from entitytrace.roster import VERSION as ROSTER_VERSION, resolve_roster
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import dumps
from tests.test_corrections import ForbiddenModels, SetupRosterModels

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RankQueryTests(unittest.TestCase):
    """rank 命令离线测试：评分公式、五级排序、章节范围、kind_rank、分页、更正联动与快照重建。

    合成夹具章节结构（正文章序只计 kind=="chapter"）：
        卷一（volume）→ 第一章 → 第二章 → 第三章 → 第四章  （K = 4）

    实体（括号内为 (提及数 m, 章节数 c, m*c)）：
        沈墨    第一章×3、第二章×1           (4, 2, 8)
        顾云    第三章×2、第四章×4           (6, 2, 12)
        青石渡  第二章×1                     (1, 1, 1)
        茶杯    第一章×1                     (1, 1, 1)
        铁剑    第一章×1                     (1, 1, 1)   — 与茶杯同 m、c、start，按 id 排序
        影客    卷一 ×1                      (1, 1, 1)   — 只在卷题区间，span_chapters=0
    """

    TEXT = (
        "卷一 风云初起\n"
        "影客在卷首现身。\n"
        "第一章 渡口\n"
        "沈墨携茶杯与铁剑上岸，沈墨又遇沈墨。\n"
        "第二章 客栈\n"
        "沈墨住在青石渡客栈。\n"
        "第三章 夜话\n"
        "顾云来访顾云。\n"
        "第四章 远行\n"
        "顾云与顾云、顾云还有顾云同行。\n"
    )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "rank.sqlite3"
        self.out_root = self.root / "outputs"
        self.run_id = self._build_fixture()
        store = Store(self.db_path)
        try:
            self.result_doc = json.loads(
                export_run(store, self.run_id, self.out_root).joinpath("result.json").read_text(encoding="utf-8")
            )
            self.meta_snap = json.loads(
                store.rows("SELECT meta_json FROM query_snapshots WHERE run_id=?", (self.run_id,))[0]["meta_json"]
            )
            self.chapters = self.meta_snap["chapters"]
            self.body_ids = [c["id"] for c in self.chapters if c.get("kind") == "chapter"]
            self.vol_ids = [c["id"] for c in self.chapters if c.get("kind") != "chapter"]
            self.by_name = {e["name"]: e for e in self.result_doc["entities"]}
        finally:
            store.close()

    def tearDown(self):
        self.temp.cleanup()

    def _build_fixture(self) -> str:
        book_path = self.root / "rank_book.txt"
        book_path.write_bytes(self.TEXT.encode("utf-8"))
        store = Store(self.db_path)
        try:
            source = import_text(store, book_path)

            def span(title: str, kind: str) -> dict:
                start = self.TEXT.index(title)
                return {"start": start, "end": start + len(title), "title": title, "kind": kind}

            ver, chapters = build_chapters(
                source,
                [
                    span("卷一 风云初起", "volume"),
                    span("第一章 渡口", "chapter"),
                    span("第二章 客栈", "chapter"),
                    span("第三章 夜话", "chapter"),
                    span("第四章 远行", "chapter"),
                ],
            )
            run_id = "run_rank_fixture_0001_abcdef123456"
            cfg = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {"complete": True, "chapter_set_version": ver, "issues": 0, "path_counts": {"adopted": 5}},
                "discovery": {"complete": True, "tasks": 5, "completed": 5},
            }
            store.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                (run_id, source["id"], dumps(cfg), dumps(state), "now", "now"),
            )
            for ch in chapters:
                store.put(run_id, "chapter", ch)

            # 每章一次 discovery：提及以（称呼, 粗类, 出现次数）声明
            disc_by_title = {
                "卷一 风云初起": [("影客", "person", 1)],
                "第一章 渡口": [
                    ("沈墨", "person", 3),
                    ("茶杯", "object", 1),
                    ("铁剑", "object", 1),
                ],
                "第二章 客栈": [("沈墨", "person", 1), ("青石渡", "place", 1)],
                "第三章 夜话": [("顾云", "person", 2)],
                "第四章 远行": [("顾云", "person", 4)],
            }
            tasks = plan_tasks(source, chapters, 9000)
            for idx, task in enumerate(tasks):
                decls = disc_by_title[task["chapter_title"]]
                unit_id = task["units"][0]["id"]
                disc = {
                    "mentions": [{"text": s, "kind": k, "refs": [unit_id] * n} for s, k, n in decls],
                    "links": [],
                }
                grounded = ground_result(source, task, disc, f"req_d{idx}")
                recs = [
                    (kind, rec)
                    for kind, key in [
                        ("mention", "mentions"),
                        ("observation", "observations"),
                        ("review", "reviews"),
                        ("link", "links"),
                    ]
                    for rec in grounded[key]
                ]
                store.finish_task(run_id, task, {"request_id": f"req_d{idx}"}, recs)
                store.call(run_id, "inv_init", "discovery", f"req_d{idx}", "discovery", 1, "success", 0.1, {"usage": {}})

            kind_by_surface = {
                "影客": "person",
                "沈墨": "person",
                "茶杯": "object",
                "铁剑": "object",
                "青石渡": "place",
                "顾云": "person",
            }

            def handle_c1(st):
                uid = st["units"][0]["id"]
                entities = []
                for i, row in enumerate(st["surfaces"]):
                    entities.append({
                        "local_id": f"e{i}",
                        "name": row["surface"],
                        "kind": kind_by_surface[row["surface"]],
                        "surfaces": [{"surface": row["surface"], "surface_type": "proper_name"}],
                        "evidence_ref": uid,
                    })
                return {"entities": entities, "assignments": [], "not_entity": []}, "req_c1_x"

            models = SetupRosterModels({
                "roster-c1": handle_c1,
                "roster-c2-review": lambda st: (
                    {
                        "groups": [{
                            "group_id": "g1",
                            "name": st["rows"][0]["name"],
                            "kind": st["rows"][0]["kind"],
                            "row_ids": [r["row_id"] for r in st["rows"]],
                            "reason": "同一对象",
                        }],
                        "unresolved_row_ids": [],
                    },
                    "req_c2_rev",
                ),
                "roster-c2-propose": lambda st: ({"proposals": []}, "req_c2_prop"),
                "roster-c3-link": lambda st: ({"links": []}, "req_c3_x"),
            })
            outcome = resolve_roster(store, models, source, run_id, concurrency=1, log=lambda _: None)
            self.assertTrue(outcome["complete"])
            store.state(run_id, identity=outcome)
            return run_id
        finally:
            store.close()

    # ---------- 工具 ----------

    def _invoke(self, argv: list[str]) -> tuple[int, str, str]:
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with patch("entitytrace.pipeline.Models", ForbiddenModels), \
             contextlib.redirect_stdout(out_buf), \
             contextlib.redirect_stderr(err_buf):
            rc = cli_main(argv)
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def _rank(self, extra: list[str]) -> tuple[int, dict, str]:
        rc, out, err = self._invoke(["rank", self.run_id, *extra, "--db", str(self.db_path)])
        doc = json.loads(out.splitlines()[0]) if out.strip() else {}
        return rc, doc, err

    def _rank_ok(self, extra: list[str]) -> dict:
        rc, doc, err = self._rank(extra)
        self.assertEqual(rc, 0, err)
        self.assertEqual(doc["protocol"], "entitytrace.v1")
        self.assertEqual(doc["command"], "rank")
        self.assertEqual(doc["run_id"], self.run_id)
        self.assertIsNone(doc["error"])
        return doc

    def _rank_err(self, extra: list[str]) -> dict:
        rc, doc, err = self._rank(extra)
        self.assertEqual(rc, 1, f"期望退出码 1，实际 doc={doc} stderr={err}")
        self.assertEqual(doc["error"]["category"], "input")
        self.assertFalse(doc["error"]["retryable"])
        return doc

    def _item(self, doc: dict, name: str) -> dict:
        return next(i for i in doc["items"] if i["name"] == name)

    # ---------- 1. 公式与五级排序 ----------

    def test_score_formula_and_five_level_sorting(self):
        doc = self._rank_ok(["--limit", "50"])
        self.assertEqual(doc["range"], {"from_chapter": 1, "to_chapter": 4, "chapter_count": 4})
        self.assertEqual(doc["total"], 6)
        items = doc["items"]
        # 第一级 score 降序：顾云 sqrt(12)=3.464103 > 沈墨 sqrt(8)=2.828427
        self.assertEqual([i["name"] for i in items[:2]], ["顾云", "沈墨"])
        gu = self._item(doc, "顾云")
        self.assertEqual(gu["mention_count_in_range"], 6)
        self.assertEqual(gu["chapter_count_in_range"], 2)
        self.assertEqual(gu["score"], round(math.sqrt(12), 6))
        self.assertEqual(gu["span_chapters"], 2)  # 正文章序 3..4
        self.assertEqual(gu["first_chapter_id"], self.body_ids[2])
        self.assertEqual(gu["last_chapter_id"], self.body_ids[3])
        shen = self._item(doc, "沈墨")
        self.assertEqual(shen["score"], round(math.sqrt(8), 6))
        self.assertEqual(shen["span_chapters"], 2)  # 正文章序 1..2
        self.assertEqual(shen["first_chapter_id"], self.body_ids[0])
        self.assertEqual(shen["last_chapter_id"], self.body_ids[1])
        # 第 3-6 名同为 m*c=1：第 4 级 min_start 升序 → 影客(卷首) < 茶杯 < 铁剑 < 青石渡(第二章)
        tail = items[2:]
        self.assertEqual([t["name"] for t in tail], ["影客", "茶杯", "铁剑", "青石渡"])
        self.assertEqual(tail[-1]["first_chapter_id"], self.body_ids[1])
        # 影客只在卷题区间：span_chapters == 0
        shadow = self._item(doc, "影客")
        self.assertEqual(shadow["span_chapters"], 0)
        self.assertEqual(shadow["mention_count_in_range"], 1)
        self.assertIn(shadow["first_chapter_id"], self.vol_ids)
        # 茶杯 min_start < 铁剑 min_start（茶杯先出现）
        cup = self._item(doc, "茶杯")
        sword = self._item(doc, "铁剑")
        cup_m = next(m for m in self.result_doc["mentions"] if m["entity_id"] == cup["id"])
        sword_m = next(m for m in self.result_doc["mentions"] if m["entity_id"] == sword["id"])
        self.assertLess(cup_m["start"], sword_m["start"])

    # ---------- 2. 章节范围过滤 ----------

    def test_chapter_range_filtering_default_partial_and_bounds(self):
        # 只给 --from-chapter：到最后一章（正文章序 3..4）
        doc = self._rank_ok(["--from-chapter", "3"])
        self.assertEqual(doc["range"], {"from_chapter": 3, "to_chapter": 4, "chapter_count": 2})
        self.assertEqual(doc["total"], 1)
        gu = self._item(doc, "顾云")
        self.assertEqual(gu["mention_count_in_range"], 6)
        self.assertEqual(gu["span_chapters"], 2)

        # 只给 --to-chapter：从第 1 章起（正文章序 1..2），卷题区间提及不计入
        doc = self._rank_ok(["--to-chapter", "2"])
        self.assertEqual(doc["range"], {"from_chapter": 1, "to_chapter": 2, "chapter_count": 2})
        names = sorted(i["name"] for i in doc["items"])
        self.assertEqual(names, ["沈墨", "茶杯", "铁剑", "青石渡"])
        shen = self._item(doc, "沈墨")
        self.assertEqual(shen["mention_count_in_range"], 4)
        self.assertEqual(shen["chapter_count_in_range"], 2)

        # N == M 单章
        doc = self._rank_ok(["--from-chapter", "2", "--to-chapter", "2"])
        self.assertEqual(doc["range"], {"from_chapter": 2, "to_chapter": 2, "chapter_count": 1})
        names = sorted(i["name"] for i in doc["items"])
        self.assertEqual(names, ["沈墨", "青石渡"])
        shen = self._item(doc, "沈墨")
        self.assertEqual(shen["mention_count_in_range"], 1)
        self.assertEqual(shen["chapter_count_in_range"], 1)
        self.assertEqual(shen["span_chapters"], 1)

        # 只出现在卷题区间的实体在显式章序范围内被排除；范围内有提及的实体其卷题提及不计 m
        doc = self._rank_ok(["--from-chapter", "1", "--to-chapter", "1"])
        names = sorted(i["name"] for i in doc["items"])
        self.assertEqual(names, ["沈墨", "茶杯", "铁剑"])

        # 越界与 N > M 均为 input、退出码 1
        for extra in (
            ["--from-chapter", "0"],
            ["--to-chapter", "0"],
            ["--from-chapter", "5"],
            ["--to-chapter", "5"],
            ["--from-chapter", "3", "--to-chapter", "1"],
            ["--from-chapter", "-2"],
        ):
            with self.subTest(extra=extra):
                self._rank_err(extra)

        # 非法分页参数
        self._rank_err(["--limit", "0"])
        self._rank_err(["--limit", "201"])
        self._rank_err(["--offset", "-1"])

    # ---------- 3. kind_rank 与 --kind ----------

    def test_kind_rank_and_kind_filter(self):
        doc = self._rank_ok(["--kind", "person", "--limit", "50"])
        self.assertEqual([i["name"] for i in doc["items"]], ["顾云", "沈墨", "影客"])
        self.assertEqual([i["kind_rank"] for i in doc["items"]], [1, 2, 3])

        doc = self._rank_ok(["--kind", "object", "--limit", "50"])
        names = [i["name"] for i in doc["items"]]
        self.assertEqual(sorted(names), ["茶杯", "铁剑"])
        self.assertEqual([i["kind_rank"] for i in doc["items"]], [1, 2])
        # 茶杯 min_start 更小，kind_rank=1
        self.assertEqual(names[0], "茶杯")

        # 不带 --kind 时 items 仍携带各自 kind 内名次
        doc = self._rank_ok(["--limit", "50"])
        rank_by_name = {i["name"]: i["kind_rank"] for i in doc["items"]}
        self.assertEqual(rank_by_name["顾云"], 1)
        self.assertEqual(rank_by_name["沈墨"], 2)
        self.assertEqual(rank_by_name["青石渡"], 1)  # place 唯一
        self.assertEqual(rank_by_name["茶杯"], 1)
        self.assertEqual(rank_by_name["铁剑"], 2)
        self.assertEqual(rank_by_name["影客"], 3)  # person 第三

        # 章序范围内重算 kind_rank
        doc = self._rank_ok(["--kind", "person", "--to-chapter", "2"])
        names = [i["name"] for i in doc["items"]]
        self.assertEqual(names, ["沈墨"])
        self.assertEqual(doc["items"][0]["kind_rank"], 1)

        # 非法 kind
        self._rank_err(["--kind", "ghost"])

    # ---------- 4. 分页 ----------

    def test_pagination_limit_offset_next_offset(self):
        doc = self._rank_ok(["--limit", "2", "--offset", "0"])
        self.assertEqual(doc["total"], 6)
        self.assertEqual(len(doc["items"]), 2)
        self.assertEqual(doc["next_offset"], 2)
        doc2 = self._rank_ok(["--limit", "2", "--offset", "4"])
        self.assertEqual(len(doc2["items"]), 2)
        self.assertIsNone(doc2["next_offset"])
        doc3 = self._rank_ok(["--limit", "2", "--offset", "2"])
        # 三页拼接覆盖全部 6 个实体且顺序一致
        full = doc["items"] + doc3["items"] + doc2["items"]
        self.assertEqual([i["id"] for i in full], [i["id"] for i in self._rank_ok(["--limit", "50"])["items"]])

    # ---------- 5. correct merge / revoke 联动 ----------

    def test_correct_merge_then_revoke_restores_ranking(self):
        rev0 = self.result_doc["result_revision"]
        before = self._rank_ok(["--kind", "object", "--limit", "50"])
        before_ids = [i["id"] for i in before["items"]]
        cup_id = self.by_name["茶杯"]["id"]
        sword_id = self.by_name["铁剑"]["id"]

        out_buf, err_buf = io.StringIO(), io.StringIO()
        with patch("entitytrace.pipeline.Models", ForbiddenModels), \
             contextlib.redirect_stdout(out_buf), \
             contextlib.redirect_stderr(err_buf):
            rc = cli_main([
                "correct", self.run_id, "merge", cup_id, sword_id,
                "--source", "user", "--expect-revision", rev0,
                "--db", str(self.db_path), "--output", str(self.out_root),
            ])
        self.assertEqual(rc, 0, err_buf.getvalue())
        receipt = json.loads(out_buf.getvalue().strip())
        rev1 = receipt["result_revision"]

        doc = self._rank_ok(["--kind", "object", "--limit", "50"])
        self.assertEqual(doc["result_revision"], rev1)
        self.assertEqual(doc["total"], 1)
        merged = doc["items"][0]
        self.assertEqual(merged["id"], cup_id)
        self.assertEqual(merged["identity_status"], "user_corrected")
        self.assertEqual(merged["mention_count_in_range"], 2)
        self.assertEqual(merged["chapter_count_in_range"], 1)
        self.assertEqual(merged["score"], round(math.sqrt(2), 6))
        self.assertEqual(merged["kind_rank"], 1)

        # 撤回
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with patch("entitytrace.pipeline.Models", ForbiddenModels), \
             contextlib.redirect_stdout(out_buf), \
             contextlib.redirect_stderr(err_buf):
            rc = cli_main([
                "correct", self.run_id, "revoke", receipt["correction_id"],
                "--source", "user", "--expect-revision", rev1,
                "--db", str(self.db_path), "--output", str(self.out_root),
            ])
        self.assertEqual(rc, 0, err_buf.getvalue())
        after = self._rank_ok(["--kind", "object", "--limit", "50"])
        self.assertEqual(after["result_revision"], rev0)
        self.assertEqual([i["id"] for i in after["items"]], before_ids)
        for new_item, old_item in zip(after["items"], before["items"]):
            self.assertEqual(new_item, old_item)

    # ---------- 6. 旧版快照自动重建 ----------

    def test_stale_snapshot_auto_rebuild(self):
        store = Store(self.db_path)
        try:
            # 情形 A：meta 去掉 snapshot_version
            row = store.rows("SELECT meta_json FROM query_snapshots WHERE run_id=?", (self.run_id,))[0]
            meta = json.loads(row["meta_json"])
            meta.pop("snapshot_version", None)
            store.execute(
                "UPDATE query_snapshots SET meta_json=? WHERE run_id=?",
                (json.dumps(meta, ensure_ascii=False, separators=(",", ":")), self.run_id),
            )
        finally:
            store.close()
        doc = self._rank_ok(["--limit", "50"])
        self.assertEqual(doc["total"], 6)
        store = Store(self.db_path)
        try:
            meta2 = json.loads(
                store.rows("SELECT meta_json FROM query_snapshots WHERE run_id=?", (self.run_id,))[0]["meta_json"]
            )
            self.assertEqual(meta2["snapshot_version"], 2)
            # 情形 B：entities 去掉 chapter_stats
            ents = json.loads(
                store.rows("SELECT entities_json FROM query_snapshots WHERE run_id=?", (self.run_id,))[0]["entities_json"]
            )
            self.assertIn("chapter_stats", ents[0])
            for e in ents:
                e.pop("chapter_stats", None)
            store.execute(
                "UPDATE query_snapshots SET entities_json=? WHERE run_id=?",
                (json.dumps(ents, ensure_ascii=False, separators=(",", ":")), self.run_id),
            )
        finally:
            store.close()
        doc = self._rank_ok(["--limit", "50"])
        self.assertEqual(doc["total"], 6)
        store = Store(self.db_path)
        try:
            ents2 = json.loads(
                store.rows("SELECT entities_json FROM query_snapshots WHERE run_id=?", (self.run_id,))[0]["entities_json"]
            )
            self.assertIn("chapter_stats", ents2[0])
        finally:
            store.close()

    # ---------- 7. 导出三件套逐字节不变 ----------

    def test_export_three_files_byte_invariant(self):
        out_dir = self.out_root / self.run_id[:16]
        before = {name: _sha256(out_dir / name) for name in ("result.json", "review.json", "manifest.json")}
        # rank 触发快照读（可能重建）后再导出，三件套必须逐字节相同
        self._rank_ok(["--limit", "50"])
        store = Store(self.db_path)
        try:
            export_run(store, self.run_id, self.out_root)
        finally:
            store.close()
        for name, sha in before.items():
            self.assertEqual(_sha256(out_dir / name), sha, f"{name} 逐字节不一致")


if __name__ == "__main__":
    unittest.main()
