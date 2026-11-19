from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.corrections import apply_correction
from entitytrace.discovery import plan_tasks
from entitytrace.pipeline import build_export_bundle, export_run
from entitytrace.propagation import resolve_redundant_grounding_reviews
from entitytrace.roster import VERSION as ROSTER_VERSION, resolve_roster
from entitytrace.storage import Store
from entitytrace.text import address, import_text
from entitytrace.util import dumps


class FakeRosterModels:
    def __init__(self, c1_handler):
        self.c1_handler = c1_handler

    def generate(self, system, state, stage, validate, *args, **kwargs):
        if stage == "roster-c1":
            val = self.c1_handler(state)
            validate(val)
            return val, f"req_{stage}_{state.get('chapter_title', 't')}"
        val = {"pairs": []} if "propose" in stage else {"groups": [], "unresolved_row_ids": []}
        validate(val)
        return val, f"req_{stage}"


def _obs(obs_id: str, task_id: str, text: str, status: str, mention_ids: list[str]) -> dict:
    return {
        "id": obs_id,
        "task_id": task_id,
        "request_id": "req_d1",
        "text": text,
        "kind": "person",
        "reported_refs": ["u_x"],
        "mention_ids": mention_ids,
        "status": status,
    }


def _review(rev_id: str, obs_id: str, status: str = "pending", stage: str = "grounding") -> dict:
    item = {
        "id": rev_id,
        "stage": stage,
        "reason": "原文引用内找不到完整称呼",
        "observation_id": obs_id,
        "status": status,
    }
    if status != "pending":
        item["resolved_by"] = "user_x"
    return item


def _mention(mid: str, source_id: str, text: str, start: int, task_id: str) -> dict:
    return {
        "id": mid,
        "text": text,
        "start": start,
        "end": start + len(text),
        "address": address(source_id, start, start + len(text)),
        "chapter_id": "ch1",
        "task_id": task_id,
        "provisional_kinds": ["person"],
        "observation_ids": [],
        "evidence": [],
        "entity_id": "e1",
        "link_method": "chapter_local",
        "link_refs": [],
    }


class ResolveRedundantGroundingReviewUnitTests(unittest.TestCase):
    """纯函数单测：A1（250 码点邻文已由相邻任务覆盖）与 B1（同记录已定位且称呼在正文内）。"""

    def _setup(self, text: str, tasks: list[tuple[int, int]]):
        source = {"id": "src_s2", "text": text}
        dtasks = [
            {"id": f"task_{i}", "stage": "discovery", "start": s, "end": e}
            for i, (s, e) in enumerate(tasks)
        ]
        return source, dtasks

    def _call(self, source, dtasks, obs, mentions, reviews):
        return resolve_redundant_grounding_reviews(
            source=source,
            discovery_tasks=dtasks,
            observations=obs,
            mentions=mentions,
            reviews=reviews,
        )

    def test_a1_resolves_when_surface_in_prev_margin_and_registered_by_adjacent_task(self):
        text = "阿岚" + ("。" * 248) + ("正文" * 30)
        source, dtasks = self._setup(text, [(250, len(text))])
        obs = _obs("obs_a", "task_0", "阿岚", "unlocated", [])
        rev = _review("rev_a", "obs_a")
        m_prev = _mention("m_prev", source["id"], "阿岚", 0, "task_other")
        updated, resolved = self._call(source, dtasks, [obs], [m_prev], [rev])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(updated[0]["status"], "resolved")
        self.assertEqual(updated[0]["resolved_by"], "auto_context_margin_covered")

    def test_a1_resolves_when_surface_in_next_margin_and_registered_by_adjacent_task(self):
        text = ("正文" * 30) + ("。" * 248) + "阿岚"
        end = len(text) - 250
        source, dtasks = self._setup(text, [(0, end)])
        obs = _obs("obs_a", "task_0", "阿岚", "unlocated", [])
        rev = _review("rev_a", "obs_a")
        m_next = _mention("m_next", source["id"], "阿岚", len(text) - 2, "task_other")
        updated, resolved = self._call(source, dtasks, [obs], [m_next], [rev])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(updated[0]["resolved_by"], "auto_context_margin_covered")

    def test_a1_keeps_pending_when_adjacent_task_has_not_registered_surface(self):
        text = "阿岚" + ("。" * 248) + ("正文" * 30)
        source, dtasks = self._setup(text, [(250, len(text))])
        obs = _obs("obs_a", "task_0", "阿岚", "unlocated", [])
        rev = _review("rev_a", "obs_a")
        updated, resolved = self._call(source, dtasks, [obs], [], [rev])
        self.assertEqual(resolved, [])
        self.assertIs(updated[0], rev)
        self.assertEqual(updated[0]["status"], "pending")

    def test_a1_keeps_pending_when_mention_beyond_margin(self):
        text = "阿岚" + ("。" * 260) + ("正文" * 30)
        source, dtasks = self._setup(text, [(262, len(text))])
        # 称呼虽仍出现在前邻文切片里（位置 0），但登记的提及在范围之外，构造 300 码点外提及
        text2 = "阿岚" + ("。" * 300) + ("正文" * 30)
        source2 = {"id": "src_s2", "text": text2}
        dtasks2 = [{"id": "task_0", "stage": "discovery", "start": 302, "end": len(text2)}]
        obs = _obs("obs_a", "task_0", "阿岚", "unlocated", [])
        rev = _review("rev_a", "obs_a")
        m_far = _mention("m_far", source2["id"], "阿岚", 0, "task_other")
        # 邻文窗口为 [52, 302)，提及起点 0 超出 → 保持 pending
        updated, resolved = self._call(source2, dtasks2, [obs], [m_far], [rev])
        self.assertEqual(resolved, [])
        self.assertEqual(updated[0]["status"], "pending")

    def test_a1_keeps_pending_when_surface_inside_task_body(self):
        text = ("正文" * 30) + "阿岚" + ("。" * 248) + "阿岚"
        # 任务正文含“阿岚”（位置 60），邻文（250 码点后）也有一处，但规则要求 surf 不在本任务正文内
        source, dtasks = self._setup(text, [(0, 62)])
        obs = _obs("obs_a", "task_0", "阿岚", "unlocated", [])
        rev = _review("rev_a", "obs_a")
        m_next = _mention("m_next", source["id"], "阿岚", len(text) - 2, "task_other")
        updated, resolved = self._call(source, dtasks, [obs], [m_next], [rev])
        self.assertEqual(resolved, [])
        self.assertEqual(updated[0]["status"], "pending")

    def test_a1_rejects_same_task_mention(self):
        text = "阿岚" + ("。" * 248) + ("正文" * 30)
        source, dtasks = self._setup(text, [(250, len(text))])
        obs = _obs("obs_a", "task_0", "阿岚", "unlocated", [])
        rev = _review("rev_a", "obs_a")
        m_same_task = _mention("m_same", source["id"], "阿岚", 0, "task_0")
        updated, resolved = self._call(source, dtasks, [obs], [m_same_task], [rev])
        self.assertEqual(resolved, [])
        self.assertEqual(updated[0]["status"], "pending")

    def test_b1_resolves_when_observation_grounded_and_surface_in_task_body(self):
        text = ("正文" * 30) + "沈墨" + ("。" * 50)
        source, dtasks = self._setup(text, [(0, len(text))])
        obs = _obs("obs_b", "task_0", "沈墨", "grounded", ["m_shen"])
        rev = _review("rev_b", "obs_b")
        updated, resolved = self._call(source, dtasks, [obs], [], [rev])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(updated[0]["status"], "resolved")
        self.assertEqual(updated[0]["resolved_by"], "auto_ref_mismatch_covered")

    def test_b1_keeps_pending_when_observation_unlocated(self):
        text = ("正文" * 30) + "沈墨" + ("。" * 50)
        source, dtasks = self._setup(text, [(0, len(text))])
        obs = _obs("obs_b", "task_0", "沈墨", "unlocated", [])
        rev = _review("rev_b", "obs_b")
        updated, resolved = self._call(source, dtasks, [obs], [], [rev])
        self.assertEqual(resolved, [])
        self.assertEqual(updated[0]["status"], "pending")

    def test_non_grounding_and_user_resolved_reviews_untouched(self):
        text = ("正文" * 30) + "沈墨" + ("。" * 50)
        source, dtasks = self._setup(text, [(0, len(text))])
        obs = _obs("obs_b", "task_0", "沈墨", "grounded", ["m_shen"])
        rev_stage = _review("rev_stage", "obs_b", stage="roster-c3")
        rev_done = _review("rev_done", "obs_b", status="resolved")
        original = [rev_stage, rev_done]
        updated, resolved = self._call(source, dtasks, [obs], [], original)
        self.assertEqual(resolved, [])
        self.assertIs(updated[0], rev_stage)
        self.assertIs(updated[1], rev_done)
        # 输入字典未被原地修改
        self.assertNotIn("resolved_by", rev_stage)
        self.assertEqual(rev_done["resolved_by"], "user_x")

    def test_missing_obs_or_task_keeps_pending(self):
        source, dtasks = self._setup("正文", [(0, 2)])
        rev_no_obs = _review("rev_x", "obs_missing")
        obs_no_task = _obs("obs_nt", "task_missing", "沈墨", "unlocated", [])
        rev_no_task = _review("rev_y", "obs_nt")
        updated, resolved = self._call(source, dtasks, [obs_no_task], [], [rev_no_obs, rev_no_task])
        self.assertEqual(resolved, [])
        self.assertEqual([u["status"] for u in updated], ["pending", "pending"])


class ExportGroundingReviewS2Tests(unittest.TestCase):
    """集成测试：build_export_bundle / export_run 幂等与更正撤回逐字节复原。"""

    def _build_run(self, root: Path):
        text = (
            "第一章\n"
            + ("沈墨与顾云同行，途中多有见闻。" * 8)
            + "阿岚在渡口回望。"
            + ("沈墨继续前行，不再回头。" * 8)
        )
        book = root / "book.txt"
        book.write_bytes(text.encode("utf-8"))
        store = Store(root / "test.sqlite3")
        try:
            source = import_text(store, book)
            ver, chapters = build_chapters(
                source, [{"start": 0, "end": 3, "title": "第一章", "kind": "chapter"}]
            )
            run_id = "run_s2_test_001"
            cfg = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {"complete": True, "chapter_set_version": ver, "issues": 0, "path_counts": {"adopted": 1}},
                "discovery": {"complete": True, "tasks": 2, "completed": 2},
            }
            store.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                (run_id, source["id"], dumps(cfg), dumps(state), "now", "now"),
            )
            for ch in chapters:
                store.put(run_id, "chapter", ch)

            planned = plan_tasks(source, chapters, 9000)
            self.assertEqual(len(planned), 1)
            task_a = planned[0]
            u0 = task_a["units"][0]
            alan_pos = text.index("阿岚")
            # 虚拟发现任务 C：其正文范围不含“阿岚”，但“阿岚”落在其前 250 码点邻文
            task_c_start = alan_pos + 30
            self.assertNotIn("阿岚", text[task_c_start:])
            self.assertIn("阿岚", text[max(0, task_c_start - 250):task_c_start])
            self.assertIn("沈墨", text[task_c_start:])
            task_c = {"id": "task_c_virtual", "stage": "discovery", "start": task_c_start, "end": len(text)}

            # 发现任务 A：沈墨已定位；另有 B1 待审（obs grounded 但曾漏报）与 A1 待审（“阿岚”在相邻任务邻文）
            shen_pos = text.index("沈墨")
            records_a = [
                (
                    "mention",
                    {
                        "id": "m_shen_1",
                        "text": "沈墨",
                        "start": shen_pos,
                        "end": shen_pos + 2,
                        "address": address(source["id"], shen_pos, shen_pos + 2),
                        "chapter_id": chapters[0]["id"],
                        "task_id": task_a["id"],
                        "provisional_kinds": ["person"],
                        "observation_ids": ["obs_shen"],
                        "evidence": [address(source["id"], u0["start"], u0["end"])],
                    },
                ),
                ("observation", _obs("obs_shen", task_a["id"], "沈墨", "grounded", ["m_shen_1"])),
                (
                    "mention",
                    {
                        "id": "m_alan_1",
                        "text": "阿岚",
                        "start": alan_pos,
                        "end": alan_pos + 2,
                        "address": address(source["id"], alan_pos, alan_pos + 2),
                        "chapter_id": chapters[0]["id"],
                        "task_id": task_a["id"],
                        "provisional_kinds": ["person"],
                        "observation_ids": ["obs_alan_loc"],
                        "evidence": [address(source["id"], u0["start"], u0["end"])],
                    },
                ),
                ("observation", _obs("obs_alan_loc", task_a["id"], "阿岚", "grounded", ["m_alan_1"])),
            ]
            store.finish_task(run_id, task_a, {"request_id": "req_d1"}, records_a)

            # 任务 C（仅 tasks 表行）：B1（obs_bel 已定位且“沈墨”在其正文内）与 A1（“阿岚”在其前邻文且已由任务 A 登记提及）
            records_c = [
                ("observation", _obs("obs_bel", task_c["id"], "沈墨", "grounded", ["m_shen_1"])),
                ("observation", _obs("obs_alan", task_c["id"], "阿岚", "unlocated", [])),
                ("review", _review("rev_b1", "obs_bel")),
                ("review", _review("rev_a1", "obs_alan")),
            ]
            store.finish_task(run_id, task_c, {"request_id": "req_d2"}, records_c)

            def handle_c1(st):
                surfs = [r["surface"] for r in st["surfaces"]]
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": surfs[0],
                            "kind": "person",
                            "surfaces": [{"surface": s, "surface_type": "proper_name"} for s in surfs],
                            "evidence_ref": u0["id"],
                        }
                    ],
                    "assignments": [],
                    "not_entity": [],
                }

            out = resolve_roster(
                store, FakeRosterModels(handle_c1), source, run_id,
                concurrency=1, log=lambda *_: None, task_chars=9000,
            )
            self.assertTrue(out["complete"])
            store.state(run_id, identity=out)
            return store, source, run_id
        except Exception:
            store.close()
            raise

    def test_export_resolves_a1_and_b1_and_reexport_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, source, run_id = self._build_run(root)
            try:
                bundle = build_export_bundle(store, run_id)
                revs = {r["id"]: r for r in bundle["review"]["items"]}
                self.assertEqual(revs["rev_b1"]["status"], "resolved")
                self.assertEqual(revs["rev_b1"]["resolved_by"], "auto_ref_mismatch_covered")
                self.assertEqual(revs["rev_a1"]["status"], "resolved")
                self.assertEqual(revs["rev_a1"]["resolved_by"], "auto_context_margin_covered")
                self.assertEqual(bundle["review"]["counts"]["resolved"], 2)
                self.assertEqual(bundle["review"]["counts"]["pending"], 0)
                self.assertEqual(bundle["result"]["status"], "completed")
                # entities / mentions 不因导出期消解而改变（补充提及允许存在，但原始提及必在）
                mention_ids = {m["id"] for m in bundle["result"]["mentions"]}
                self.assertIn("m_alan_1", mention_ids)
                self.assertIn("m_shen_1", mention_ids)
                self.assertTrue(all(m["link_method"] in {"chapter_local", "same_task_proper"} for m in bundle["result"]["mentions"]))
                # 存储中的待审记录仍是 pending（不污染数据库）
                raw_revs = {r["id"]: r for r in store.records(run_id, "review")}
                self.assertEqual(raw_revs["rev_a1"]["status"], "pending")
                self.assertEqual(raw_revs["rev_b1"]["status"], "pending")

                out_dir1 = export_run(store, run_id, root / "out1")
                out_dir2 = export_run(store, run_id, root / "out2")
                for name in ("result.json", "review.json", "manifest.json", "decisions.jsonl", "source.txt"):
                    self.assertEqual(
                        (out_dir1 / name).read_bytes(),
                        (out_dir2 / name).read_bytes(),
                        f"{name} 重导出应逐字节幂等",
                    )
            finally:
                store.close()

    def test_correction_then_revoke_restores_byte_identical_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "out"
            store, source, run_id = self._build_run(root)
            try:
                before = build_export_bundle(store, run_id)
                before_rev = before["result"]["result_revision"]
                # 叠加用户更正：把 m_shen_1 指派为新实体
                out_dir = apply_correction(
                    store,
                    run_id,
                    op="assign",
                    op_args={"mention_id": "m_shen_1", "target": "new"},
                    source="user",
                    expect_revision=before_rev,
                    output_root=out_root,
                )
                corrected = build_export_bundle(store, run_id)
                self.assertNotEqual(corrected["result"]["result_revision"], before_rev)
                corr_id = out_dir["correction_id"] if isinstance(out_dir, dict) and "correction_id" in out_dir else None
                # apply_correction 返回回执字典；撤回该更正
                receipt = out_dir
                self.assertIn("correction_id", receipt)
                revoked = apply_correction(
                    store,
                    run_id,
                    op="revoke",
                    op_args={"correction_id": receipt["correction_id"]},
                    source="user",
                    expect_revision=corrected["result"]["result_revision"],
                    output_root=out_root,
                )
                self.assertEqual(revoked["status"], "revoked")
                restored = build_export_bundle(store, run_id)
                self.assertEqual(restored["result"]["result_revision"], before_rev)
                revs = {r["id"]: r for r in restored["review"]["items"]}
                self.assertEqual(revs["rev_a1"]["resolved_by"], "auto_context_margin_covered")
                self.assertEqual(revs["rev_b1"]["resolved_by"], "auto_ref_mismatch_covered")

                out_a = export_run(store, run_id, root / "out_a")
                out_b = export_run(store, run_id, root / "out_b")
                for name in ("result.json", "review.json", "manifest.json"):
                    self.assertEqual((out_a / name).read_bytes(), (out_b / name).read_bytes())
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
