from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.cli import main as cli_main
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.pipeline import export_run, run_pipeline
from entitytrace.roster import VERSION as ROSTER_VERSION, resolve_roster
from entitytrace.storage import Store
from entitytrace.text import import_text
from entitytrace.util import digest, dumps

ROOT = Path(__file__).resolve().parents[1]


class ForbiddenModels:
    """一旦在更正或重导出阶段触发任何模型调用即直接失败。"""

    def __init__(self, *args, **kwargs):
        raise AssertionError("更正闭环严禁实例化 Models")

    def generate(self, *args, **kwargs):
        raise AssertionError("更正闭环严禁调用 generate")

    def choices(self, *args, **kwargs):
        raise AssertionError("更正闭环严禁调用 choices")


class SetupRosterModels:
    def __init__(self, handlers):
        self.handlers = handlers

    def choices(self, *args, **kwargs):
        raise AssertionError("roster 初始化不应调用 choices")

    def generate(self, system, state, stage, validate, *args, **kwargs):
        val, req_id = self.handlers[stage](state)
        validate(val)
        return val, req_id


def _legacy_roster_export_bytes(store: Store, run_id: str) -> tuple[bytes, bytes, bytes]:
    """按阶段 G 之前的零更正逻辑独立构造字节快照，锁定零更正输出逐字节不变。"""
    from entitytrace.pipeline import metrics
    from entitytrace.text import validate_span

    run = store.run(run_id)
    source = store.source(run["source_id"])
    text, metadata = source["text"], source["metadata"]
    chapters = store.records(run_id, "chapter")
    mentions = store.records(run_id, "mention")
    observations = store.records(run_id, "observation")
    decisions = store.records(run_id, "decision")
    reviews = store.records(run_id, "review")
    by_chapter = {c["id"]: c for c in chapters}
    state = run["state"]
    complete = all(state.get(name, {}).get("complete", False) for name in ("chapters", "discovery", "identity"))
    roster_out = store.records(run_id, "roster_output")[0]
    entities = [dict(e) for e in roster_out["entities"]]
    mentions_out = [dict(m) for m in roster_out["mentions"]]
    non_identity_reviews = [r for r in reviews if not str(r.get("stage", "")).startswith("roster")]
    identity_reviews = [r for r in roster_out["reviews"] if str(r.get("stage", "")).startswith("roster")]
    all_reviews = identity_reviews + non_identity_reviews
    for mention in mentions_out:
        validate_span(text, mention["address"], source["id"], mention["text"])
        chapter = by_chapter[mention["chapter_id"]]
        mention["chapter_start"] = mention["start"] - chapter["start"]
        mention["chapter_end"] = mention["end"] - chapter["start"]
    status = "incomplete" if not complete else "completed_with_review" if all_reviews else "completed"
    result = {
        "schema_version": 3,
        "run_id": run_id,
        "status": status,
        "source": metadata,
        "chapters": chapters,
        "mentions": mentions_out,
        "observations": observations,
        "entities": entities,
        "roster_summary": roster_out["roster_summary"],
        "ambiguous_surfaces": roster_out["ambiguous_surfaces"],
        "scope": "实体提及与名册链接身份归组；不含描述、服饰或经历提取",
        "limitations": [
            "名册条目 ID 由成员首处地址与强称呼集合摘要生成，不是跨版本永久编号。",
            "发现阶段未覆盖的提及不进入名册归组；原文定位通过不代表实体和身份语义正确。",
            "无比较或未决的单独提及（unresolved_singleton）仅表示证据不足，不能据此断言与所有其他实体不同。",
        ],
    }
    result["result_revision"] = digest(result)
    review_doc = {
        "schema_version": 1,
        "run_id": run_id,
        "result_revision": result["result_revision"],
        "items": all_reviews,
        "counts": {
            "total": len(all_reviews),
            "identity_reviews": len(identity_reviews),
            "grounding_discovery_reviews": len(non_identity_reviews),
        },
        "instruction": "按 impact 降序排列；自动模型判断不冒充人工答案。",
    }
    invocations = store.records(run_id, "invocation")
    identity_seconds = round(
        sum(
            inv.get("stages", {}).get("identity", {}).get("seconds", 0.0)
            for inv in invocations
            if not inv.get("stages", {}).get("identity", {}).get("resumed", False)
        ),
        4,
    )
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "result_revision": result["result_revision"],
        "status": status,
        "source": metadata,
        "config": run["config"],
        "state": state,
        "invocations": invocations,
        "identity_seconds": identity_seconds,
        "total_model_metrics": metrics(store, run_id),
        "counts": {
            "chapters": sum(c["kind"] == "chapter" for c in chapters),
            "sections": len(chapters),
            "chapter_path_counts": state.get("chapters", {}).get("path_counts", {}),
            "mentions": len(mentions_out),
            "observations": len(observations),
            "entities": len(entities),
            "singletons": sum(1 for e in entities if len(e["mention_ids"]) == 1),
            "unlocated_observations": sum(o["status"] == "unlocated" for o in observations),
            "decisions": len(decisions),
            "review_items": len(all_reviews),
            "identity_review_items": len(identity_reviews),
            "grounding_discovery_review_items": len(non_identity_reviews),
        },
        "validation": {
            "source_hashes": True,
            "mention_addresses": True,
            "entity_membership": True,
            "semantic_accuracy_measured": False,
        },
        "files": ["result.json", "review.json", "decisions.jsonl", "requests.jsonl", "source.txt", "source.original"],
    }
    to_bytes = lambda obj: (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return to_bytes(result), to_bytes(review_doc), to_bytes(manifest)


class RosterCorrectionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "roster.sqlite3"
        self.out_root = self.root / "outputs"
        self.run_id = self._build_synthetic_roster_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def _build_synthetic_roster_fixture(self) -> str:
        """构造完全虚构的两章合成夹具：覆盖专名组、含 c3_model 提及组、待合并组、平局粗类提及、未决待审提及与不受影响地点实体。"""
        text = (
            "第一章\n"
            "沈墨与阿墨在青石渡会见顾云和周掌柜，沈墨见那位剑客把茶杯放在桌上。\n"
            "第二章\n"
            "青石渡起了雾，神秘来客独自登船。\n"
        )
        book_path = self.root / "synthetic.txt"
        book_path.write_bytes(text.encode("utf-8"))
        store = Store(self.db_path)
        try:
            source = import_text(store, book_path)
            idx1, idx2 = text.index("第一章"), text.index("第二章")
            ver, chapters = build_chapters(
                source,
                [
                    {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                    {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
                ],
            )
            run_id = "run_corr_fixture_0001_abcdef123456"
            cfg = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {"complete": True, "chapter_set_version": ver, "issues": 0, "path_counts": {"adopted": 2}},
                "discovery": {"complete": True, "tasks": 2, "completed": 2},
            }
            store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], dumps(cfg), dumps(state), "now", "now"))
            for ch in chapters:
                store.put(run_id, "chapter", ch)

            tasks = plan_tasks(source, chapters, 9000)
            u1, u2 = tasks[0]["units"][0]["id"], tasks[1]["units"][0]["id"]
            disc1 = {
                "mentions": [
                    {"text": "沈墨", "kind": "person", "refs": [u1]},
                    # 故意让“阿墨”带 place 与 organization 两个粗类（1:1 平局），验证拆分新实体按 KINDS 排序取 organization
                    {"text": "阿墨", "kind": "place", "refs": [u1]},
                    {"text": "阿墨", "kind": "organization", "refs": [u1]},
                    {"text": "青石渡", "kind": "place", "refs": [u1]},
                    {"text": "顾云", "kind": "person", "refs": [u1]},
                    {"text": "周掌柜", "kind": "person", "refs": [u1]},
                    {"text": "那位剑客", "kind": "person", "refs": [u1]},
                    {"text": "茶杯", "kind": "object", "refs": [u1]},
                ],
                "links": [],
            }
            disc2 = {
                "mentions": [
                    {"text": "青石渡", "kind": "place", "refs": [u2]},
                    {"text": "神秘来客", "kind": "person", "refs": [u2]},
                ],
                "links": [],
            }
            for task, disc, req_id in zip(tasks, [disc1, disc2], ["req_d1", "req_d2"]):
                grounded = ground_result(source, task, disc, req_id)
                recs = [
                    (kind, rec)
                    for kind, key in [("mention", "mentions"), ("observation", "observations"), ("review", "reviews"), ("link", "links")]
                    for rec in grounded[key]
                ]
                store.finish_task(run_id, task, {"request_id": req_id}, recs)
                store.call(run_id, "inv_init", "discovery", req_id, "discovery", 1, "success", 0.1, {"usage": {}})

            def handle_c1(st):
                uid = st["units"][0]["id"]
                if st["chapter_title"] == "第一章":
                    swordsman_mid = next(r["mentions"][0]["mention_id"] for r in st["surfaces"] if r["surface"] == "那位剑客")
                    return {
                        "entities": [
                            {
                                "local_id": "e1",
                                "name": "沈墨",
                                "kind": "person",
                                "surfaces": [
                                    {"surface": "沈墨", "surface_type": "proper_name"},
                                    {"surface": "阿墨", "surface_type": "proper_name"},
                                ],
                                "evidence_ref": uid,
                            },
                            {
                                "local_id": "e2",
                                "name": "青石渡",
                                "kind": "place",
                                "surfaces": [{"surface": "青石渡", "surface_type": "proper_name"}],
                                "evidence_ref": uid,
                            },
                            {
                                "local_id": "e3",
                                "name": "顾云",
                                "kind": "person",
                                "surfaces": [{"surface": "顾云", "surface_type": "proper_name"}],
                                "evidence_ref": uid,
                            },
                            {
                                "local_id": "e4",
                                "name": "周掌柜",
                                "kind": "person",
                                "surfaces": [{"surface": "周掌柜", "surface_type": "proper_name"}],
                                "evidence_ref": uid,
                            },
                            {
                                "local_id": "e5",
                                "name": "茶杯",
                                "kind": "object",
                                "surfaces": [{"surface": "茶杯", "surface_type": "proper_name"}],
                                "evidence_ref": uid,
                            },
                        ],
                        "assignments": [
                            {"mention_id": swordsman_mid, "surface": "那位剑客", "ref": uid, "local_id": None, "reason": "待全书链接"},
                        ],
                        "not_entity": [],
                    }, "req_c1_1"
                guest_mid = next(r["mentions"][0]["mention_id"] for r in st["surfaces"] if r["surface"] == "神秘来客")
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": "青石渡",
                            "kind": "place",
                            "surfaces": [{"surface": "青石渡", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                    ],
                    "assignments": [
                        {"mention_id": guest_mid, "surface": "神秘来客", "ref": uid, "local_id": None, "reason": "身份未明"},
                    ],
                    "not_entity": [],
                }, "req_c1_2"

            def handle_c3(st):
                shen_id = next(c["entity_id"] for c in st["candidates"] if c["name"] == "沈墨")
                links = []
                for it in st["items"]:
                    if it["surface"] == "那位剑客":
                        links.append({"mention_id": it["mention_id"], "selected_entity_id": shen_id, "reason": "同指沈墨"})
                    else:
                        links.append({"mention_id": it["mention_id"], "selected_entity_id": "uncertain", "reason": "无法判断"})
                return {"links": links}, "req_c3_1"

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
                "roster-c3-link": handle_c3,
            })
            outcome = resolve_roster(store, models, source, run_id, concurrency=1, log=lambda _: None)
            self.assertTrue(outcome["complete"])
            store.state(run_id, identity=outcome)
            store.call(run_id, "inv_init", "roster-c1", "req_c1_1", "discovery", 1, "success", 0.1, {"usage": {}})
            export_run(store, run_id, self.out_root)
            return run_id
        finally:
            store.close()

    def _calls_count(self, db_path: Path | None = None) -> int:
        store = Store(db_path or self.db_path)
        try:
            return len(store.rows("SELECT * FROM calls"))
        finally:
            store.close()

    def _run_correct_cli(self, args: list[str], db_path: Path | None = None, out_root: Path | None = None) -> tuple[int, str, str]:
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        full_args = [
            "correct",
            *args,
            "--db", str(db_path or self.db_path),
            "--output", str(out_root or self.out_root),
        ]
        with patch("entitytrace.pipeline.Models", ForbiddenModels), \
             contextlib.redirect_stdout(stdout_buf), \
             contextlib.redirect_stderr(stderr_buf):
            rc = cli_main(full_args)
        return rc, stdout_buf.getvalue().strip(), stderr_buf.getvalue().strip()

    def _read_exported(self) -> tuple[dict, dict, dict, Path]:
        out_dir = self.out_root / self.run_id[:16]
        res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        rev = json.loads((out_dir / "review.json").read_text(encoding="utf-8"))
        man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        return res, rev, man, out_dir

    def test_zero_corrections_export_byte_for_byte_identical(self):
        store = Store(self.db_path)
        try:
            legacy_res, legacy_rev, legacy_man = _legacy_roster_export_bytes(store, self.run_id)
            out_dir = export_run(store, self.run_id, self.out_root)
        finally:
            store.close()
        self.assertEqual((out_dir / "result.json").read_bytes(), legacy_res)
        self.assertEqual((out_dir / "review.json").read_bytes(), legacy_rev)
        self.assertEqual((out_dir / "manifest.json").read_bytes(), legacy_man)
        res_doc = json.loads(legacy_res)
        rev_doc = json.loads(legacy_rev)
        self.assertNotIn("corrections", res_doc)
        self.assertNotIn("agent_suggestions", rev_doc)

    def test_four_ops_semantics_ids_unaffected_invariance_c3_review_resolves_and_reverse_revoke(self):
        initial_calls = self._calls_count()
        _, _, _, out_dir = self._read_exported()
        init_res_bytes = (out_dir / "result.json").read_bytes()
        init_rev_bytes = (out_dir / "review.json").read_bytes()
        init_man_bytes = (out_dir / "manifest.json").read_bytes()

        res0, rev0, _, _ = self._read_exported()
        rev_hash_0 = res0["result_revision"]
        self.assertEqual(res0["status"], "completed_with_review")
        self.assertEqual(len(rev0["items"]), 1)
        init_review_id = rev0["items"][0]["id"]

        by_name = {e["name"]: e for e in res0["entities"]}
        qingshi_before_dump = dumps(by_name["青石渡"])
        shen_id = by_name["沈墨"]["id"]
        gu_id = by_name["顾云"]["id"]
        zhou_id = by_name["周掌柜"]["id"]
        cup_mention = next(m for m in res0["mentions"] if m["text"] == "茶杯")
        guest_mention = next(m for m in res0["mentions"] if m["text"] == "神秘来客")
        swordsman_mention = next(m for m in res0["mentions"] if m["text"] == "那位剑客")
        self.assertEqual(swordsman_mention["link_method"], "c3_model")

        # 1) merge：周掌柜并入顾云，保留顾云 ID，周掌柜消失
        rc1, out1, err1 = self._run_correct_cli([
            self.run_id, "merge", gu_id, zhou_id,
            "--source", "user", "--expect-revision", rev_hash_0, "--note", "同指一人",
        ])
        self.assertEqual(rc1, 0, err1)
        rec1 = json.loads(out1)
        self.assertEqual(rec1["protocol"], "entitytrace.v1")
        self.assertEqual(rec1["command"], "correct")
        self.assertEqual(rec1["run_id"], self.run_id)
        self.assertIsNone(rec1["error"])
        self.assertEqual(rec1["status"], "applied")
        self.assertEqual(sorted(rec1["affected_entity_ids"]), sorted([gu_id, zhou_id]))
        self.assertEqual(self._calls_count(), initial_calls)

        res1, _, _, _ = self._read_exported()
        e_map1 = {e["id"]: e for e in res1["entities"]}
        self.assertIn(gu_id, e_map1)
        self.assertNotIn(zhou_id, e_map1)
        self.assertEqual(e_map1[gu_id]["identity_status"], "user_corrected")
        self.assertEqual(e_map1[gu_id]["kind"], "person")
        self.assertEqual(e_map1[gu_id]["aliases"], ["周掌柜", "顾云"])
        self.assertEqual(e_map1[gu_id]["mention_count"], 2)
        zhou_m1 = next(m for m in res1["mentions"] if m["text"] == "周掌柜")
        self.assertEqual(zhou_m1["entity_id"], gu_id)
        self.assertEqual(zhou_m1["link_method"], "user_correction")
        self.assertIn(rec1["correction_id"], zhou_m1["link_refs"])
        # 未受影响实体青石渡字典逐字节不变
        self.assertEqual(dumps(e_map1[by_name["青石渡"]["id"]]), qingshi_before_dump)

        # 2) split-surface：从沈墨拆出“阿墨”，保留沈墨 ID，生成新实体 digest(["correction-split", cid])，
        #    并为仍留在沈墨中的 c3_model 提及“那位剑客”生成 correction-c3-affected 待审项
        #    同时测试无效 split-surface 被拒绝且不写记录
        bad_rc_missing, bad_out_missing, _ = self._run_correct_cli([
            self.run_id, "split-surface", shen_id, "不存在称呼",
            "--source", "user", "--expect-revision", res1["result_revision"],
        ])
        self.assertEqual(bad_rc_missing, 1)
        self.assertEqual(json.loads(bad_out_missing)["error"]["category"], "input")
        bad_rc_all, bad_out_all, _ = self._run_correct_cli([
            self.run_id, "split-surface", by_name["青石渡"]["id"], "青石渡",
            "--source", "user", "--expect-revision", res1["result_revision"],
        ])
        self.assertEqual(bad_rc_all, 1)
        self.assertEqual(json.loads(bad_out_all)["error"]["category"], "input")

        rc2, out2, err2 = self._run_correct_cli([
            self.run_id, "split-surface", shen_id, "阿墨",
            "--source", "user", "--expect-revision", res1["result_revision"],
        ])
        self.assertEqual(rc2, 0, err2)
        rec2 = json.loads(out2)
        cid2 = rec2["correction_id"]
        expected_split_id = digest(["correction-split", cid2])
        self.assertEqual(sorted(rec2["affected_entity_ids"]), sorted([shen_id, expected_split_id]))
        res2, rev2, _, _ = self._read_exported()
        e_map2 = {e["id"]: e for e in res2["entities"]}
        self.assertIn(shen_id, e_map2)
        self.assertIn(expected_split_id, e_map2)
        self.assertEqual(e_map2[shen_id]["aliases"], ["沈墨"])
        self.assertEqual(e_map2[expected_split_id]["aliases"], ["阿墨"])
        # “阿墨”带有 place 与 organization 平局 provisional_kinds，按 KINDS 字母序取第一个 organization
        self.assertEqual(e_map2[expected_split_id]["kind"], "organization")
        # “那位剑客”仍留在沈墨中，未移动，且产生 correction-c3-affected 待审项
        sw_m2 = next(m for m in res2["mentions"] if m["id"] == swordsman_mention["id"])
        self.assertEqual(sw_m2["entity_id"], shen_id)
        self.assertEqual(sw_m2["link_method"], "c3_model")
        expected_c3_rev_id = digest([cid2, swordsman_mention["id"]])
        c3_rev_item = next(r for r in rev2["items"] if r["id"] == expected_c3_rev_id)
        self.assertEqual(c3_rev_item["stage"], "correction-c3-affected")
        self.assertEqual(c3_rev_item["status"], "pending")
        self.assertEqual(dumps(e_map2[by_name["青石渡"]["id"]]), qingshi_before_dump)

        # 3) assign（至 new 并带 --resolves 解决初始待审项；另测试未知待审项被拒绝）
        bad_rev_rc, bad_rev_out, _ = self._run_correct_cli([
            self.run_id, "assign", guest_mention["id"], "new",
            "--source", "user", "--expect-revision", res2["result_revision"],
            "--resolves", "nonexistent_review_id",
        ])
        self.assertEqual(bad_rev_rc, 1)
        self.assertEqual(json.loads(bad_rev_out)["error"]["category"], "input")

        rc3, out3, err3 = self._run_correct_cli([
            self.run_id, "assign", guest_mention["id"], "new",
            "--source", "user", "--expect-revision", res2["result_revision"],
            "--resolves", init_review_id,
        ])
        self.assertEqual(rc3, 0, err3)
        rec3 = json.loads(out3)
        cid3 = rec3["correction_id"]
        expected_new_id = digest(["correction-new", cid3])
        res3, rev3, _, _ = self._read_exported()
        e_map3 = {e["id"]: e for e in res3["entities"]}
        # 原暂定单项实体被移空删除，替换为 expected_new_id
        self.assertNotIn(guest_mention["entity_id"], e_map3)
        self.assertIn(expected_new_id, e_map3)
        self.assertEqual(e_map3[expected_new_id]["identity_status"], "user_corrected")
        resolved_item = next(r for r in rev3["items"] if r["id"] == init_review_id)
        self.assertEqual(resolved_item["status"], "resolved")
        self.assertEqual(resolved_item["resolved_by"], cid3)

        # 4) not-entity（同时带 --resolves 解决刚才的 c3 受影响待审项，使全部 pending 待审清零，status 变为 completed）
        rc4, out4, err4 = self._run_correct_cli([
            self.run_id, "not-entity", cup_mention["id"],
            "--source", "user", "--expect-revision", res3["result_revision"],
            "--resolves", expected_c3_rev_id,
        ])
        self.assertEqual(rc4, 0, err4)
        rec4 = json.loads(out4)
        cid4 = rec4["correction_id"]
        expected_ne_id = digest(["correction-not-entity", cup_mention["id"]])
        res4, rev4, _, _ = self._read_exported()
        e_map4 = {e["id"]: e for e in res4["entities"]}
        self.assertIn(expected_ne_id, e_map4)
        self.assertEqual(e_map4[expected_ne_id]["identity_status"], "not_entity_confirmed")
        cup_m4 = next(m for m in res4["mentions"] if m["id"] == cup_mention["id"])
        self.assertEqual(cup_m4["entity_id"], expected_ne_id)
        self.assertEqual(cup_m4["identity_status"], "not_entity_confirmed")
        self.assertEqual(cup_m4["link_method"], "user_correction")
        # 实体恰好覆盖全部提及
        all_mids = sorted(m["id"] for m in res4["mentions"])
        covered_mids = sorted(mid for e in res4["entities"] for mid in e["mention_ids"])
        self.assertEqual(covered_mids, all_mids)
        # 所有待审均已 resolved -> status 转为 completed
        self.assertEqual(res4["status"], "completed")
        self.assertEqual(len(res4["corrections"]), 4)
        self.assertIn("含用户更正，被更正实体的 ID 规则见 corrections", res4["limitations"])
        self.assertEqual(dumps(e_map4[by_name["青石渡"]["id"]]), qingshi_before_dump)
        self.assertEqual(self._calls_count(), initial_calls)

        # 5) 逆序逐条撤回，撤回后 result.json / review.json / manifest.json 逐字节完全复原
        cur_rev = res4["result_revision"]
        for target_cid in [cid4, cid3, cid2, rec1["correction_id"]]:
            rc_rev, out_rev, err_rev = self._run_correct_cli([
                self.run_id, "revoke", target_cid,
                "--source", "user", "--expect-revision", cur_rev,
            ])
            self.assertEqual(rc_rev, 0, err_rev)
            rev_receipt = json.loads(out_rev)
            self.assertEqual(rev_receipt["status"], "revoked")
            cur_rev = rev_receipt["result_revision"]

        self.assertEqual(cur_rev, rev_hash_0)
        self.assertEqual((out_dir / "result.json").read_bytes(), init_res_bytes)
        self.assertEqual((out_dir / "review.json").read_bytes(), init_rev_bytes)
        self.assertEqual((out_dir / "manifest.json").read_bytes(), init_man_bytes)
        self.assertEqual(self._calls_count(), initial_calls)

    def test_agent_suggestion_stale_revision_idempotency_and_revoke_permissions(self):
        initial_calls = self._calls_count()
        res0, _, _, out_dir = self._read_exported()
        init_res_bytes = (out_dir / "result.json").read_bytes()
        init_rev_bytes = (out_dir / "review.json").read_bytes()
        rev0_hash = res0["result_revision"]
        by_name = {e["name"]: e for e in res0["entities"]}
        gu_id, zhou_id = by_name["顾云"]["id"], by_name["周掌柜"]["id"]

        # 1) 版本过期拒绝且不写任何记录
        rc_stale, out_stale, err_stale = self._run_correct_cli([
            self.run_id, "merge", gu_id, zhou_id,
            "--source", "user", "--expect-revision", "stale_wrong_revision",
        ])
        self.assertEqual(rc_stale, 1)
        stale_env = json.loads(out_stale)
        self.assertEqual(stale_env["protocol"], "entitytrace.v1")
        self.assertEqual(stale_env["command"], "correct")
        self.assertEqual(stale_env["run_id"], self.run_id)
        self.assertEqual(stale_env["error"]["category"], "conflict")
        self.assertFalse(stale_env["error"]["retryable"])
        self.assertIn("结果版本已变化，请重新读取", err_stale)
        store = Store(self.db_path)
        try:
            self.assertEqual(store.corrections(self.run_id), [])
        finally:
            store.close()

        # 2) agent 建议不改变 result.json 和版本，但写入 review.json 的 agent_suggestions
        rc_ag, out_ag, err_ag = self._run_correct_cli([
            self.run_id, "merge", gu_id, zhou_id,
            "--source", "agent", "--expect-revision", rev0_hash, "--note", "模型推测同指",
        ])
        self.assertEqual(rc_ag, 0, err_ag)
        ag_receipt = json.loads(out_ag)
        self.assertEqual(ag_receipt["status"], "suggested")
        self.assertEqual(ag_receipt["result_revision"], rev0_hash)
        self.assertEqual(ag_receipt["affected_entity_ids"], [])
        self.assertEqual((out_dir / "result.json").read_bytes(), init_res_bytes)
        _, rev_after_ag, _, _ = self._read_exported()
        self.assertEqual(len(rev_after_ag["agent_suggestions"]), 1)
        self.assertEqual(rev_after_ag["agent_suggestions"][0]["id"], ag_receipt["correction_id"])

        # 3) agent 重复提交同一建议 -> 幂等返回 duplicate，不新增记录
        rc_ag_dup, out_ag_dup, _ = self._run_correct_cli([
            self.run_id, "merge", gu_id, zhou_id,
            "--source", "agent", "--expect-revision", rev0_hash,
        ])
        self.assertEqual(rc_ag_dup, 0)
        ag_dup_receipt = json.loads(out_ag_dup)
        self.assertEqual(ag_dup_receipt["status"], "duplicate")
        self.assertEqual(ag_dup_receipt["correction_id"], ag_receipt["correction_id"])

        # 4) user 提交同一 merge -> 生效；再次重复提交 -> duplicate
        rc_usr, out_usr, _ = self._run_correct_cli([
            self.run_id, "merge", gu_id, zhou_id,
            "--source", "user", "--expect-revision", rev0_hash,
        ])
        self.assertEqual(rc_usr, 0)
        usr_receipt = json.loads(out_usr)
        self.assertEqual(usr_receipt["status"], "applied")
        rev1_hash = usr_receipt["result_revision"]
        self.assertNotEqual(rev1_hash, rev0_hash)

        rc_usr_dup, out_usr_dup, _ = self._run_correct_cli([
            self.run_id, "merge", gu_id, zhou_id,
            "--source", "user", "--expect-revision", rev1_hash,
        ])
        self.assertEqual(rc_usr_dup, 0)
        usr_dup_receipt = json.loads(out_usr_dup)
        self.assertEqual(usr_dup_receipt["status"], "duplicate")
        self.assertEqual(usr_dup_receipt["correction_id"], usr_receipt["correction_id"])
        self.assertEqual(usr_dup_receipt["result_revision"], rev1_hash)

        # 5) 权限校验：agent 撤回 user 更正被拒绝；agent 撤回自身建议成功；user 撤回 user 更正成功；重复撤回幂等
        rc_forbid, out_forbid, err_forbid = self._run_correct_cli([
            self.run_id, "revoke", usr_receipt["correction_id"],
            "--source", "agent", "--expect-revision", rev1_hash,
        ])
        self.assertEqual(rc_forbid, 1)
        self.assertEqual(json.loads(out_forbid)["error"]["category"], "input")
        self.assertIn("agent 只能撤回 agent 建议", err_forbid)

        rc_rev_ag, out_rev_ag, _ = self._run_correct_cli([
            self.run_id, "revoke", ag_receipt["correction_id"],
            "--source", "agent", "--expect-revision", rev1_hash,
        ])
        self.assertEqual(rc_rev_ag, 0)
        self.assertEqual(json.loads(out_rev_ag)["status"], "revoked")

        rc_rev_usr, out_rev_usr, _ = self._run_correct_cli([
            self.run_id, "revoke", usr_receipt["correction_id"],
            "--source", "user", "--expect-revision", rev1_hash,
        ])
        self.assertEqual(rc_rev_usr, 0)
        rev_usr_receipt = json.loads(out_rev_usr)
        self.assertEqual(rev_usr_receipt["status"], "revoked")
        self.assertEqual(rev_usr_receipt["result_revision"], rev0_hash)

        # 重复撤回同一 ID -> duplicate
        rc_rev_dup, out_rev_dup, _ = self._run_correct_cli([
            self.run_id, "revoke", usr_receipt["correction_id"],
            "--source", "user", "--expect-revision", rev0_hash,
        ])
        self.assertEqual(rc_rev_dup, 0)
        self.assertEqual(json.loads(out_rev_dup)["status"], "duplicate")
        self.assertEqual((out_dir / "result.json").read_bytes(), init_res_bytes)
        self.assertEqual((out_dir / "review.json").read_bytes(), init_rev_bytes)
        self.assertEqual(self._calls_count(), initial_calls)

    def test_dependent_chain_revoke_requires_cascade_and_restores_byte_for_byte(self):
        res0, _, _, out_dir = self._read_exported()
        init_res_bytes = (out_dir / "result.json").read_bytes()
        init_rev_bytes = (out_dir / "review.json").read_bytes()
        init_man_bytes = (out_dir / "manifest.json").read_bytes()
        by_name = {e["name"]: e for e in res0["entities"]}
        shen_id = by_name["沈墨"]["id"]
        zhou_m = next(m for m in res0["mentions"] if m["text"] == "周掌柜")

        # 第一步：split-surface 产生新实体 split_id
        rc1, out1, _ = self._run_correct_cli([
            self.run_id, "split-surface", shen_id, "阿墨",
            "--source", "user", "--expect-revision", res0["result_revision"],
        ])
        self.assertEqual(rc1, 0)
        r1 = json.loads(out1)
        cid1 = r1["correction_id"]
        split_id = digest(["correction-split", cid1])

        # 第二步：将“周掌柜”提及 assign 到第一步新建的 split_id（依赖 cid1）
        rc2, out2, _ = self._run_correct_cli([
            self.run_id, "assign", zhou_m["id"], split_id,
            "--source", "user", "--expect-revision", r1["result_revision"],
        ])
        self.assertEqual(rc2, 0)
        r2 = json.loads(out2)
        cid2 = r2["correction_id"]

        # 第三步：从 split_id 再把“周掌柜” split-surface 出来（依赖 cid2）
        rc3, out3, _ = self._run_correct_cli([
            self.run_id, "split-surface", split_id, "周掌柜",
            "--source", "user", "--expect-revision", r2["result_revision"],
        ])
        self.assertEqual(rc3, 0)
        r3 = json.loads(out3)
        cid3 = r3["correction_id"]

        # 不加 --cascade 撤回 cid1：必须拒绝，并在错误信息中列出依赖它的 cid2 和 cid3，且不新增记录
        store = Store(self.db_path)
        try:
            count_before = len(store.corrections(self.run_id))
        finally:
            store.close()
        rc_fail, out_fail, err_fail = self._run_correct_cli([
            self.run_id, "revoke", cid1,
            "--source", "user", "--expect-revision", r3["result_revision"],
        ])
        self.assertEqual(rc_fail, 1)
        self.assertEqual(json.loads(out_fail)["error"]["category"], "input")
        self.assertIn(cid2, err_fail)
        self.assertIn(cid3, err_fail)
        store = Store(self.db_path)
        try:
            self.assertEqual(len(store.corrections(self.run_id)), count_before)
        finally:
            store.close()

        # 加上 --cascade 撤回 cid1：一并撤回 cid1、cid2、cid3，三个导出文件逐字节复原
        rc_ok, out_ok, err_ok = self._run_correct_cli([
            self.run_id, "revoke", cid1, "--cascade",
            "--source", "user", "--expect-revision", r3["result_revision"],
        ])
        self.assertEqual(rc_ok, 0, err_ok)
        ok_receipt = json.loads(out_ok)
        self.assertEqual(ok_receipt["status"], "revoked")
        self.assertEqual(ok_receipt["result_revision"], res0["result_revision"])
        self.assertEqual((out_dir / "result.json").read_bytes(), init_res_bytes)
        self.assertEqual((out_dir / "review.json").read_bytes(), init_rev_bytes)
        self.assertEqual((out_dir / "manifest.json").read_bytes(), init_man_bytes)

    def test_groups_v2_and_incomplete_runs_rejected(self):
        from tests.test_mvp import FakeModels

        text = self.root / "mini.txt"
        text.write_text("第一章 初见\n林舟来了。\n第二章 再见\n林舟走了。\n", encoding="utf-8")
        cfg = self.root / ".env"
        cfg.write_text("TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")

        # 1) groups-v2 完成运行 -> 拒绝
        v2_db = self.root / "v2.sqlite3"
        v2_out = self.root / "v2_out"
        FakeModels.fail_once = False
        with patch("entitytrace.pipeline.Models", FakeModels):
            v2_run_id, v2_dir, v2_ok = run_pipeline(
                text, v2_db, v2_out, cfg, identity_algorithm="groups-v2", log=lambda _: None,
            )
        self.assertTrue(v2_ok)
        v2_rev = json.loads((v2_dir / "result.json").read_text(encoding="utf-8"))["result_revision"]
        v2_ents = json.loads((v2_dir / "result.json").read_text(encoding="utf-8"))["entities"]
        rc_v2, out_v2, err_v2 = self._run_correct_cli(
            [v2_run_id, "not-entity", v2_ents[0]["mention_ids"][0], "--source", "user", "--expect-revision", v2_rev],
            db_path=v2_db, out_root=v2_out,
        )
        self.assertEqual(rc_v2, 1)
        self.assertEqual(json.loads(out_v2)["error"]["category"], "input")
        self.assertIn("entity-roster-v1", err_v2)

        # 2) roster 未完成运行（stop_after="discovery"） -> 拒绝
        inc_db = self.root / "inc.sqlite3"
        inc_out = self.root / "inc_out"
        with patch("entitytrace.pipeline.Models", FakeModels):
            inc_run_id, inc_dir, inc_ok = run_pipeline(
                text, inc_db, inc_out, cfg, stop_after="discovery", log=lambda _: None,
            )
        self.assertFalse(inc_ok)
        inc_res = json.loads((inc_dir / "result.json").read_text(encoding="utf-8"))
        rc_inc, out_inc, err_inc = self._run_correct_cli(
            [inc_run_id, "not-entity", inc_res["mentions"][0]["id"], "--source", "user", "--expect-revision", inc_res["result_revision"]],
            db_path=inc_db, out_root=inc_out,
        )
        self.assertEqual(rc_inc, 1)
        self.assertEqual(json.loads(out_inc)["error"]["category"], "input")
        self.assertIn("尚未完成", err_inc)

    def test_query_snapshot_immediately_reflects_correct_and_revoke(self):
        from entitytrace.query import (
            query_entities,
            query_entity,
            query_quote_json,
            query_review,
            query_status,
        )

        res0, rev0, _, _ = self._read_exported()
        rev0_hash = res0["result_revision"]
        init_review_id = rev0["items"][0]["id"]
        by_name = {e["name"]: e for e in res0["entities"]}
        gu_id = by_name["顾云"]["id"]
        zhou_id = by_name["周掌柜"]["id"]
        guest_mention = next(m for m in res0["mentions"] if m["text"] == "神秘来客")

        store = Store(self.db_path)
        try:
            st0 = query_status(store, self.run_id, self.out_root)
            ents0 = query_entities(store, self.run_id)
            ent_gu0 = query_entity(store, self.run_id, gu_id)
            revs0 = query_review(store, self.run_id)
            qt0 = query_quote_json(store, self.run_id, 0, 4)
        finally:
            store.close()
        self.assertEqual(st0["result_revision"], rev0_hash)
        self.assertEqual(ents0["result_revision"], rev0_hash)
        self.assertEqual(ent_gu0["result_revision"], rev0_hash)
        self.assertEqual(ent_gu0["mention_count"], 1)
        self.assertEqual(revs0["result_revision"], rev0_hash)
        self.assertEqual(revs0["total"], 1)
        self.assertEqual(qt0["result_revision"], rev0_hash)

        rc1, out1, err1 = self._run_correct_cli([
            self.run_id, "merge", gu_id, zhou_id,
            "--source", "user", "--expect-revision", rev0_hash,
            "--resolves", init_review_id,
        ])
        self.assertEqual(rc1, 0, err1)
        rec1 = json.loads(out1)
        rev1_hash = rec1["result_revision"]
        self.assertNotEqual(rev1_hash, rev0_hash)

        # 直接在未重新 export 的数据库上也可以验证查询快照立即反映最新更正
        store = Store(self.db_path)
        try:
            st1 = query_status(store, self.run_id, self.out_root)
            ents1 = query_entities(store, self.run_id)
            ent_gu1 = query_entity(store, self.run_id, gu_id)
            revs1 = query_review(store, self.run_id)
            qt1 = query_quote_json(store, self.run_id, guest_mention["start"], guest_mention["end"])
            with self.assertRaises(ValueError):
                query_entity(store, self.run_id, zhou_id)
        finally:
            store.close()
        self.assertEqual(st1["result_revision"], rev1_hash)
        self.assertEqual(st1["status"], "completed")
        self.assertEqual(st1["counts"]["entities"], st0["counts"]["entities"] - 1)
        self.assertEqual(st1["counts"]["review_items"], 0)
        self.assertEqual(ents1["result_revision"], rev1_hash)
        self.assertEqual(ents1["total"], ents0["total"] - 1)
        self.assertEqual(ent_gu1["result_revision"], rev1_hash)
        self.assertEqual(ent_gu1["mention_count"], 2)
        self.assertEqual(ent_gu1["aliases"], ["周掌柜", "顾云"])
        self.assertEqual(revs1["result_revision"], rev1_hash)
        self.assertEqual(revs1["total"], 0)
        self.assertEqual(qt1["result_revision"], rev1_hash)

        rc_rev, out_rev, err_rev = self._run_correct_cli([
            self.run_id, "revoke", rec1["correction_id"],
            "--source", "user", "--expect-revision", rev1_hash,
        ])
        self.assertEqual(rc_rev, 0, err_rev)
        store = Store(self.db_path)
        try:
            st2 = query_status(store, self.run_id, self.out_root)
            ents2 = query_entities(store, self.run_id)
            ent_gu2 = query_entity(store, self.run_id, gu_id)
            revs2 = query_review(store, self.run_id)
            qt2 = query_quote_json(store, self.run_id, 0, 4)
        finally:
            store.close()
        self.assertEqual(st2, st0)
        self.assertEqual(ents2, ents0)
        self.assertEqual(ent_gu2, ent_gu0)
        self.assertEqual(revs2, revs0)
        self.assertEqual(qt2, qt0)

    def test_corrections_module_isolation_no_models_no_gold_no_book_names(self):
        src = (ROOT / "entitytrace/corrections.py").read_text(encoding="utf-8")
        for forbidden in ["entity-gold", "evaluate_identity_gold", "Pride", "九命奇冤", "伤逝", "Second Stain", "Elizabeth", "Darcy"]:
            self.assertNotIn(forbidden, src)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith("tools"))
                self.assertNotIn("models", node.module)


if __name__ == "__main__":
    unittest.main()
