"""生产回归测试：合成提及用户更正覆盖不变量与描述全缓存复跑墙钟幂等。"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from entitytrace.chapters import build_chapters
from entitytrace.corrections import _validate_replayed_coverage, apply_correction
from entitytrace.describe import _run_persistence, run_describe
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.models import ModelError
from entitytrace.pipeline import build_export_bundle, export_run
from entitytrace.roster import VERSION as ROSTER_VERSION, resolve_roster
from entitytrace.storage import Store
from entitytrace.text import address, import_text
from entitytrace.util import dumps

from tests.test_describe import FakeModels, FixtureCase, SERVICES


class _SetupRosterModels:
    def __init__(self, handlers):
        self.handlers = handlers

    def choices(self, *args, **kwargs):
        raise AssertionError("名册初始化不应调用 choices")

    def generate(self, system, state, stage, validate, *args, **kwargs):
        val, req_id = self.handlers[stage](state)
        validate(val)
        return val, req_id


class _NoCallModels:
    """全缓存复跑期间一旦触发模型调用即直接失败。"""

    def __init__(self, services):
        self.services = services

    def generate(self, *args, **kwargs):
        raise AssertionError("全缓存复跑严禁发起任何 generate 调用")

    def choices(self, *args, **kwargs):
        raise AssertionError("全缓存复跑严禁发起任何 choices 调用")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _monotonic_pairs(*elapsed_values: float):
    """为每次 run_describe(begin, end) 构造精确可控的 monotonic 序列。"""
    seq: list[float] = []
    cur = 100.0
    for dt in elapsed_values:
        seq.append(cur)
        cur += dt
        seq.append(cur)
        cur += 1.0
    it = iter(seq)
    return lambda: next(it)


class SyntheticMentionCorrectionTests(unittest.TestCase):
    """覆盖含 same_task_proper 合成提及的实体更正、写库前预检与逆序撤回。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "synthetic_corr.sqlite3"
        self.out_root = self.root / "outputs"
        self.run_id = self._build_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def _build_fixture(self) -> str:
        # 第一章故意让“东院”出现两次，但发现只报第一处，触发同任务专名补全生成 same_task_proper 合成提及；
        # “青荷”同样出现两次，发现只报第一处，构成另一实体中的合成提及用于 merge 测试。
        text = (
            "第一章\n"
            "沈墨住在西院，又常去东院拜访。后来东院修了门楼，青荷带着青荷的旧箱子来西院做客。\n"
            "第二章\n"
            "西院门前起了风，沈墨出门远行。\n"
        )
        book = self.root / "novel.txt"
        book.write_bytes(text.encode("utf-8"))
        store = Store(self.db_path)
        try:
            source = import_text(store, book)
            i1, i2 = text.index("第一章"), text.index("第二章")
            ver, chapters = build_chapters(
                source,
                [
                    {"start": i1, "end": i1 + 3, "title": "第一章", "kind": "chapter"},
                    {"start": i2, "end": i2 + 3, "title": "第二章", "kind": "chapter"},
                ],
            )
            run_id = "run_synth_corr_019c_000000000001"
            cfg = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
            state = {
                "chapters": {"complete": True, "chapter_set_version": ver, "issues": 0, "path_counts": {"adopted": 2}},
                "discovery": {"complete": True, "tasks": 2, "completed": 2},
            }
            store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, source["id"], dumps(cfg), dumps(state), "now", "now"))
            for ch in chapters:
                store.put(run_id, "chapter", ch)

            tasks = plan_tasks(source, chapters, 9000)
            # 手工写入发现提及：每个目标专名在第一章只放首处，留第二处给 propagate_same_task_proper 补全
            ch1_id, ch2_id = chapters[0]["id"], chapters[1]["id"]
            u1 = tasks[0]["units"][0]["id"]
            u2 = tasks[1]["units"][0]["id"]

            def _make_mention(mid: str, surface: str, kind: str, ch_id: str, task: dict, start: int):
                end = start + len(surface)
                u0 = task["units"][0]
                return {
                    "id": mid,
                    "text": surface,
                    "start": start,
                    "end": end,
                    "address": address(source["id"], start, end),
                    "chapter_id": ch_id,
                    "task_id": task["id"],
                    "provisional_kinds": [kind],
                    "observation_ids": [f"obs_{mid}"],
                    "evidence": [address(source["id"], u0["start"], u0["end"])],
                }

            ch1_mentions = [
                _make_mention("m_shen_1", "沈墨", "person", ch1_id, tasks[0], text.index("沈墨")),
                _make_mention("m_xiyuan_1", "西院", "place", ch1_id, tasks[0], text.index("西院")),
                _make_mention("m_dongyuan_1", "东院", "place", ch1_id, tasks[0], text.index("东院")),
                _make_mention("m_qinghe_1", "青荷", "person", ch1_id, tasks[0], text.index("青荷")),
            ]
            ch2_start = i2
            ch2_mentions = [
                _make_mention("m_xiyuan_2", "西院", "place", ch2_id, tasks[1], text.index("西院", ch2_start)),
                _make_mention("m_shen_2", "沈墨", "person", ch2_id, tasks[1], text.index("沈墨", ch2_start)),
            ]
            store.finish_task(run_id, tasks[0], {"request_id": "req_d1"}, [("mention", m) for m in ch1_mentions])
            store.finish_task(run_id, tasks[1], {"request_id": "req_d2"}, [("mention", m) for m in ch2_mentions])

            def handle_c1(st):
                uid = st["units"][0]["id"]
                if st["chapter_title"] == "第一章":
                    # 故意把“西院”和“东院”合在同一实体，模拟双专名误并拆分场景
                    return {
                        "entities": [
                            {
                                "local_id": "e1",
                                "name": "沈墨",
                                "kind": "person",
                                "surfaces": [{"surface": "沈墨", "surface_type": "proper_name"}],
                                "evidence_ref": uid,
                            },
                            {
                                "local_id": "e2",
                                "name": "西院",
                                "kind": "place",
                                "surfaces": [
                                    {"surface": "西院", "surface_type": "proper_name"},
                                    {"surface": "东院", "surface_type": "proper_name"},
                                ],
                                "evidence_ref": uid,
                            },
                            {
                                "local_id": "e3",
                                "name": "青荷",
                                "kind": "person",
                                "surfaces": [{"surface": "青荷", "surface_type": "proper_name"}],
                                "evidence_ref": uid,
                            },
                        ],
                        "assignments": [],
                        "not_entity": [],
                    }, "req_c1_1"
                return {
                    "entities": [
                        {
                            "local_id": "e1",
                            "name": "西院",
                            "kind": "place",
                            "surfaces": [{"surface": "西院", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                        {
                            "local_id": "e2",
                            "name": "沈墨",
                            "kind": "person",
                            "surfaces": [{"surface": "沈墨", "surface_type": "proper_name"}],
                            "evidence_ref": uid,
                        },
                    ],
                    "assignments": [],
                    "not_entity": [],
                }, "req_c1_2"

            def handle_c2_review(st):
                return {
                    "groups": [{
                        "group_id": "g1",
                        "name": st["rows"][0]["name"],
                        "kind": st["rows"][0]["kind"],
                        "row_ids": [r["row_id"] for r in st["rows"]],
                        "reason": "同一对象",
                    }],
                    "unresolved_row_ids": [],
                }, "req_c2_rev"

            def handle_c2_propose(st):
                return {"proposals": []}, "req_c2_prop"

            models = _SetupRosterModels({
                "roster-c1": handle_c1,
                "roster-c2-review": handle_c2_review,
                "roster-c2-propose": handle_c2_propose,
            })
            out = resolve_roster(store, models, source, run_id, concurrency=1, log=lambda *_: None, task_chars=9000)
            self.assertTrue(out["complete"])
            store.state(run_id, identity=out)

            bundle = build_export_bundle(store, run_id)
            raw_ids = {m["id"] for m in store.records(run_id, "mention")}
            synth = [m for m in bundle["result"]["mentions"] if m["id"] not in raw_ids]
            # “东院”第二处与“青荷”第二处（以及第一章第二处“西院”）应已由 same_task_proper 补全
            self.assertGreaterEqual(len(synth), 2)
            self.assertTrue(any(m["text"] == "东院" for m in synth))
            self.assertTrue(any(m["text"] == "青荷" for m in synth))
            self.assertTrue(all(m["link_method"] == "same_task_proper" for m in synth))
            return run_id
        finally:
            store.close()

    def _entity_by_name(self, bundle: dict, name: str) -> dict:
        return next(e for e in bundle["result"]["entities"] if e["name"] == name)

    def _synthetic_mentions(self, store: Store, bundle: dict) -> list[dict]:
        raw_ids = {m["id"] for m in store.records(self.run_id, "mention")}
        return [m for m in bundle["result"]["mentions"] if m["id"] not in raw_ids]

    def test_split_surface_touching_synthetic_mentions_and_revoke_roundtrip(self):
        store = Store(self.db_path)
        try:
            out_dir = export_run(store, self.run_id, self.out_root)
            before_sha = _file_sha256(out_dir / "result.json")
            before = build_export_bundle(store, self.run_id)
            rev0 = before["result"]["result_revision"]
            xiyuan = self._entity_by_name(before, "西院")

            r1 = apply_correction(
                store, self.run_id, "split-surface",
                {"entity_id": xiyuan["id"], "surface": "东院"},
                source="user", expect_revision=rev0, output_root=self.out_root,
            )
            self.assertEqual(r1["status"], "applied")
            after = build_export_bundle(store, self.run_id)
            synth_dong = [m for m in self._synthetic_mentions(store, after) if m["text"] == "东院"]
            self.assertEqual(len(synth_dong), 1)
            self.assertEqual(synth_dong[0]["link_method"], "user_correction")

            r_rev = apply_correction(
                store, self.run_id, "revoke",
                {"correction_id": r1["correction_id"]},
                source="user", expect_revision=r1["result_revision"], output_root=self.out_root,
            )
            self.assertEqual(r_rev["status"], "revoked")
            self.assertEqual(r_rev["result_revision"], rev0)
            self.assertEqual(_file_sha256(out_dir / "result.json"), before_sha)
        finally:
            store.close()

    def test_merge_entity_containing_synthetic_mentions_and_revoke_roundtrip(self):
        store = Store(self.db_path)
        try:
            out_dir = export_run(store, self.run_id, self.out_root)
            before_sha = _file_sha256(out_dir / "result.json")
            before = build_export_bundle(store, self.run_id)
            rev0 = before["result"]["result_revision"]
            shen = self._entity_by_name(before, "沈墨")
            qinghe = self._entity_by_name(before, "青荷")

            r1 = apply_correction(
                store, self.run_id, "merge",
                {"target_entity_id": shen["id"], "merged_entity_id": qinghe["id"]},
                source="user", expect_revision=rev0, output_root=self.out_root,
            )
            self.assertEqual(r1["status"], "applied")
            after = build_export_bundle(store, self.run_id)
            synth_qh = [m for m in self._synthetic_mentions(store, after) if m["text"] == "青荷"]
            self.assertEqual(len(synth_qh), 1)
            self.assertEqual(synth_qh[0]["entity_id"], shen["id"])
            self.assertEqual(synth_qh[0]["link_method"], "user_correction")

            r_rev = apply_correction(
                store, self.run_id, "revoke",
                {"correction_id": r1["correction_id"]},
                source="user", expect_revision=r1["result_revision"], output_root=self.out_root,
            )
            self.assertEqual(r_rev["result_revision"], rev0)
            self.assertEqual(_file_sha256(out_dir / "result.json"), before_sha)
        finally:
            store.close()

    def test_assign_synthetic_mention_to_existing_and_new_entity_and_revoke(self):
        store = Store(self.db_path)
        try:
            out_dir = export_run(store, self.run_id, self.out_root)
            before_sha = _file_sha256(out_dir / "result.json")
            before = build_export_bundle(store, self.run_id)
            rev0 = before["result"]["result_revision"]
            shen = self._entity_by_name(before, "沈墨")
            synth_dong = next(m for m in self._synthetic_mentions(store, before) if m["text"] == "东院")
            synth_qh = next(m for m in self._synthetic_mentions(store, before) if m["text"] == "青荷")

            r1 = apply_correction(
                store, self.run_id, "assign",
                {"mention_id": synth_dong["id"], "target": shen["id"]},
                source="user", expect_revision=rev0, output_root=self.out_root,
            )
            r2 = apply_correction(
                store, self.run_id, "assign",
                {"mention_id": synth_qh["id"], "target": "new"},
                source="user", expect_revision=r1["result_revision"], output_root=self.out_root,
            )
            after = build_export_bundle(store, self.run_id)
            by_id = {m["id"]: m for m in after["result"]["mentions"]}
            self.assertEqual(by_id[synth_dong["id"]]["link_method"], "user_correction")
            self.assertEqual(by_id[synth_dong["id"]]["entity_id"], shen["id"])
            self.assertEqual(by_id[synth_qh["id"]]["link_method"], "user_correction")

            r_rev2 = apply_correction(
                store, self.run_id, "revoke",
                {"correction_id": r2["correction_id"]},
                source="user", expect_revision=r2["result_revision"], output_root=self.out_root,
            )
            r_rev1 = apply_correction(
                store, self.run_id, "revoke",
                {"correction_id": r1["correction_id"]},
                source="user", expect_revision=r_rev2["result_revision"], output_root=self.out_root,
            )
            self.assertEqual(r_rev1["result_revision"], rev0)
            self.assertEqual(_file_sha256(out_dir / "result.json"), before_sha)
        finally:
            store.close()

    def test_not_entity_on_synthetic_mention_and_revoke_roundtrip(self):
        store = Store(self.db_path)
        try:
            out_dir = export_run(store, self.run_id, self.out_root)
            before_sha = _file_sha256(out_dir / "result.json")
            before = build_export_bundle(store, self.run_id)
            rev0 = before["result"]["result_revision"]
            synth_dong = next(m for m in self._synthetic_mentions(store, before) if m["text"] == "东院")

            r1 = apply_correction(
                store, self.run_id, "not-entity",
                {"mention_id": synth_dong["id"]},
                source="user", expect_revision=rev0, output_root=self.out_root,
            )
            after = build_export_bundle(store, self.run_id)
            updated = next(m for m in after["result"]["mentions"] if m["id"] == synth_dong["id"])
            self.assertEqual(updated["link_method"], "user_correction")
            self.assertEqual(updated["identity_status"], "not_entity_confirmed")

            r_rev = apply_correction(
                store, self.run_id, "revoke",
                {"correction_id": r1["correction_id"]},
                source="user", expect_revision=r1["result_revision"], output_root=self.out_root,
            )
            self.assertEqual(r_rev["result_revision"], rev0)
            self.assertEqual(_file_sha256(out_dir / "result.json"), before_sha)
        finally:
            store.close()

    def test_chained_split_and_not_entity_on_synthetic_mentions_reverse_revoke(self):
        store = Store(self.db_path)
        try:
            out_dir = export_run(store, self.run_id, self.out_root)
            before_sha = _file_sha256(out_dir / "result.json")
            before = build_export_bundle(store, self.run_id)
            rev = before["result"]["result_revision"]
            xiyuan = self._entity_by_name(before, "西院")

            cids: list[str] = []
            r_split = apply_correction(
                store, self.run_id, "split-surface",
                {"entity_id": xiyuan["id"], "surface": "东院"},
                source="user", expect_revision=rev, output_root=self.out_root,
            )
            cids.append(r_split["correction_id"])
            rev = r_split["result_revision"]

            mid_bundle = build_export_bundle(store, self.run_id)
            dong_ent = self._entity_by_name(mid_bundle, "东院")
            self.assertEqual(len(dong_ent["mention_ids"]), 2)
            for mid in list(dong_ent["mention_ids"]):
                r_ne = apply_correction(
                    store, self.run_id, "not-entity",
                    {"mention_id": mid},
                    source="user", expect_revision=rev, output_root=self.out_root,
                )
                cids.append(r_ne["correction_id"])
                rev = r_ne["result_revision"]

            for cid in reversed(cids):
                r_rev = apply_correction(
                    store, self.run_id, "revoke",
                    {"correction_id": cid},
                    source="user", expect_revision=rev, output_root=self.out_root,
                )
                rev = r_rev["result_revision"]
            self.assertEqual(rev, before["result"]["result_revision"])
            self.assertEqual(_file_sha256(out_dir / "result.json"), before_sha)
        finally:
            store.close()

    def test_no_active_user_rejects_tampered_user_correction_on_synthetic_mention(self):
        store = Store(self.db_path)
        try:
            raw_ids = {m["id"] for m in store.records(self.run_id, "mention")}
            roster_out = dict(store.records(self.run_id, "roster_output")[0])
            tampered = [dict(m) for m in roster_out["mentions"]]
            changed = 0
            for m in tampered:
                if m["id"] not in raw_ids:
                    m["link_method"] = "user_correction"
                    changed += 1
                    break
            self.assertEqual(changed, 1)
            roster_out["mentions"] = tampered
            store.put(self.run_id, "roster_output", roster_out)
            with self.assertRaises(ValueError):
                build_export_bundle(store, self.run_id)
        finally:
            store.close()

    def test_active_user_still_rejects_illegal_link_method_on_synthetic_mention(self):
        store = Store(self.db_path)
        try:
            before = build_export_bundle(store, self.run_id)
            xiyuan = self._entity_by_name(before, "西院")
            apply_correction(
                store, self.run_id, "split-surface",
                {"entity_id": xiyuan["id"], "surface": "东院"},
                source="user", expect_revision=before["result"]["result_revision"], output_root=self.out_root,
            )
            raw_ids = {m["id"] for m in store.records(self.run_id, "mention")}
            roster_out = dict(store.records(self.run_id, "roster_output")[0])
            tampered = [dict(m) for m in roster_out["mentions"]]
            for m in tampered:
                if m["id"] not in raw_ids and m["text"] == "青荷":
                    m["link_method"] = "book_proper"
            roster_out["mentions"] = tampered
            store.put(self.run_id, "roster_output", roster_out)
            with self.assertRaises(ValueError):
                build_export_bundle(store, self.run_id)
        finally:
            store.close()

    def test_prewrite_validation_blocks_corrupted_replay_without_polluting_corrections(self):
        store = Store(self.db_path)
        try:
            before = build_export_bundle(store, self.run_id)
            rev0 = before["result"]["result_revision"]
            xiyuan = self._entity_by_name(before, "西院")
            raw_ids = {m["id"] for m in store.records(self.run_id, "mention")}

            # 模拟回放产物出现非法合成提及 link_method：预检必须在 store.append_correction 之前拦截
            real_replay = __import__("entitytrace.corrections", fromlist=["replay_user_corrections"]).replay_user_corrections

            def corrupted_replay(*args, **kwargs):
                res = real_replay(*args, **kwargs)
                mutated = [dict(m) for m in res["mentions"]]
                for m in mutated:
                    if m["id"] not in raw_ids:
                        m["link_method"] = "c3_model"
                        break
                res["mentions"] = mutated
                return res

            with patch("entitytrace.corrections.replay_user_corrections", side_effect=corrupted_replay):
                with self.assertRaises(ValueError):
                    apply_correction(
                        store, self.run_id, "split-surface",
                        {"entity_id": xiyuan["id"], "surface": "东院"},
                        source="user", expect_revision=rev0, output_root=self.out_root,
                    )
            self.assertEqual(store.corrections(self.run_id), [])
        finally:
            store.close()

    def test_validate_replayed_coverage_direct_invariants(self):
        raw_ids = {"m1"}
        base_entities = [{"id": "e1", "mention_ids": ["m1", "m_syn"]}]
        syn_same = [{"id": "m1", "link_method": "c1"}, {"id": "m_syn", "link_method": "same_task_proper"}]
        syn_user = [{"id": "m1", "link_method": "c1"}, {"id": "m_syn", "link_method": "user_correction"}]
        syn_bad = [{"id": "m1", "link_method": "c1"}, {"id": "m_syn", "link_method": "book_proper"}]

        _validate_replayed_coverage(raw_ids, base_entities, syn_same, has_active_user=False)
        _validate_replayed_coverage(raw_ids, base_entities, syn_user, has_active_user=True)
        with self.assertRaises(ValueError):
            _validate_replayed_coverage(raw_ids, base_entities, syn_user, has_active_user=False)
        with self.assertRaises(ValueError):
            _validate_replayed_coverage(raw_ids, base_entities, syn_bad, has_active_user=True)
        with self.assertRaises(ValueError):
            _validate_replayed_coverage({"m1", "m_missing"}, base_entities, syn_same, has_active_user=True)
        with self.assertRaises(ValueError):
            _validate_replayed_coverage(raw_ids, [{"id": "e1", "mention_ids": ["m1"]}], syn_same, has_active_user=True)


class DescribeReplayWallIdempotencyTests(FixtureCase):
    """覆盖 run_describe 全缓存零新任务复跑保留首次 wall_seconds 与新任务刷新边界。"""

    def _verify_and_persist_handler(self, stage, state, call_index):
        if stage == "describe-verify":
            return {
                "verdicts": [
                    {"id": item["id"], "value": item["value"], "verdict": "keep", "reason": "ok"}
                    for item in state["items"]
                ]
            }
        if stage == "describe-event-verify":
            return {
                "verdicts": [
                    {"id": item["id"], "value": item["value"], "verdict": "keep", "stage_split": False, "reason": "ok"}
                    for item in state["items"]
                ]
            }
        if stage == "describe-persistence":
            return {"items": [{"id": item["id"], "label": "trait"} for item in state["items"]]}
        return self._handler(stage, state, call_index)

    def test_describe_replay_persistence_zero_raw_bytes_identical(self):
        out1 = self.root / "desc_p0_1.json"
        out2 = self.root / "desc_p0_2.json"
        models1 = FakeModels(SERVICES, self._verify_and_persist_handler)
        models2 = _NoCallModels(SERVICES)
        store = Store(self.db_path)
        try:
            with patch("entitytrace.describe.time.monotonic", side_effect=_monotonic_pairs(12.3456, 0.05)):
                r1 = run_describe(
                    store, self.run_id, services=SERVICES, models=models1,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=1,
                    persistence=0, out=out1,
                )
                r2 = run_describe(
                    store, self.run_id, services=SERVICES, models=models2,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=1,
                    persistence=0, out=out2,
                )
            self.assertEqual(r1["describe_id"], r2["describe_id"])
            self.assertEqual(out1.read_bytes(), out2.read_bytes())
            self.assertEqual(_file_sha256(out1), _file_sha256(out2))
            self.assertEqual(r2["document"]["stats"]["wall_seconds"], 12.3456)
            self.assertEqual(set(r2["document"]["stats"]), {"requests", "prompt_tokens", "completion_tokens", "wall_seconds"})
            self.assertEqual(r2["replay_wall_seconds"], 0.05)
        finally:
            store.close()

    def test_describe_replay_persistence_one_raw_bytes_identical(self):
        out1 = self.root / "desc_p1_1.json"
        out2 = self.root / "desc_p1_2.json"
        models1 = FakeModels(SERVICES, self._verify_and_persist_handler)
        models2 = _NoCallModels(SERVICES)
        store = Store(self.db_path)
        try:
            with patch("entitytrace.describe.time.monotonic", side_effect=_monotonic_pairs(8.5, 0.02)):
                r1 = run_describe(
                    store, self.run_id, services=SERVICES, models=models1,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=1,
                    persistence=1, out=out1,
                )
                r2 = run_describe(
                    store, self.run_id, services=SERVICES, models=models2,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=1,
                    persistence=1, out=out2,
                )
            self.assertEqual(r1["describe_id"], r2["describe_id"])
            self.assertEqual(out1.read_bytes(), out2.read_bytes())
            self.assertEqual(r2["document"]["stats"]["wall_seconds"], 8.5)
            self.assertEqual(r2["replay_wall_seconds"], 0.02)
            self.assertEqual(set(r2["document"]["stats"]), {"requests", "prompt_tokens", "completion_tokens", "wall_seconds"})
        finally:
            store.close()

    def test_describe_triple_consecutive_replays_preserve_first_wall(self):
        out1 = self.root / "desc_t1.json"
        out2 = self.root / "desc_t2.json"
        out3 = self.root / "desc_t3.json"
        models1 = FakeModels(SERVICES, self._verify_and_persist_handler)
        store = Store(self.db_path)
        try:
            with patch("entitytrace.describe.time.monotonic", side_effect=_monotonic_pairs(5.25, 0.04, 0.09)):
                run_describe(
                    store, self.run_id, services=SERVICES, models=models1,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=0, out=out1,
                )
                r2 = run_describe(
                    store, self.run_id, services=SERVICES, models=_NoCallModels(SERVICES),
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=0, out=out2,
                )
                r3 = run_describe(
                    store, self.run_id, services=SERVICES, models=_NoCallModels(SERVICES),
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=0, out=out3,
                )
            self.assertEqual(_file_sha256(out1), _file_sha256(out2))
            self.assertEqual(_file_sha256(out1), _file_sha256(out3))
            self.assertEqual(r2["document"]["stats"]["wall_seconds"], 5.25)
            self.assertEqual(r3["document"]["stats"]["wall_seconds"], 5.25)
            self.assertEqual(r3["replay_wall_seconds"], 0.09)
        finally:
            store.close()

    def test_describe_partial_resume_with_new_d1_task_updates_wall_seconds(self):
        models1 = FakeModels(SERVICES, self._verify_and_persist_handler)
        models2 = FakeModels(SERVICES, self._verify_and_persist_handler)
        store = Store(self.db_path)
        try:
            with patch("entitytrace.describe.time.monotonic", side_effect=_monotonic_pairs(10.0, 3.25)):
                r1 = run_describe(
                    store, self.run_id, services=SERVICES, models=models1,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=0,
                )
                # 删掉一个 D1 任务模拟未完结断点续跑
                d1_task_id = store.rows(
                    "SELECT id FROM describe_tasks WHERE describe_id=? LIMIT 1",
                    (r1["describe_id"],),
                )[0]["id"]
                store.execute(
                    "DELETE FROM describe_tasks WHERE describe_id=? AND id=?",
                    (r1["describe_id"], d1_task_id),
                )
                r2 = run_describe(
                    store, self.run_id, services=SERVICES, models=models2,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=0,
                )
            self.assertGreater(r2["new_tasks"], 0)
            self.assertEqual(r2["document"]["stats"]["wall_seconds"], 3.25)
        finally:
            store.close()

    def test_describe_resume_with_new_persistence_task_updates_wall_seconds(self):
        models1 = FakeModels(SERVICES, self._verify_and_persist_handler)
        models2 = FakeModels(SERVICES, self._verify_and_persist_handler)
        store = Store(self.db_path)
        try:
            with patch("entitytrace.describe.time.monotonic", side_effect=_monotonic_pairs(9.9, 2.75)):
                r1 = run_describe(
                    store, self.run_id, services=SERVICES, models=models1,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=1,
                )
                store.execute(
                    "DELETE FROM describe_tasks WHERE describe_id=? AND id LIKE 'persistence-%'",
                    (r1["describe_id"],),
                )
                r2 = run_describe(
                    store, self.run_id, services=SERVICES, models=models2,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=1,
                )
            self.assertEqual(r2["new_tasks"], 0)
            self.assertEqual(r2["document"]["stats"]["wall_seconds"], 2.75)
        finally:
            store.close()

    def test_describe_persistence_invalid_output_counts_as_new_task(self):
        def failing_persist_handler(stage, state, call_index):
            if stage == "describe-persistence":
                raise ModelError("invalid-output")
            return self._verify_and_persist_handler(stage, state, call_index)

        models1 = FakeModels(SERVICES, self._verify_and_persist_handler)
        models2 = FakeModels(SERVICES, failing_persist_handler)
        store = Store(self.db_path)
        try:
            with patch("entitytrace.describe.time.monotonic", side_effect=_monotonic_pairs(7.5, 1.5)):
                r1 = run_describe(
                    store, self.run_id, services=SERVICES, models=models1,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=1,
                )
                store.execute(
                    "DELETE FROM describe_tasks WHERE describe_id=? AND id LIKE 'persistence-%'",
                    (r1["describe_id"],),
                )
                r2 = run_describe(
                    store, self.run_id, services=SERVICES, models=models2,
                    top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                    persistence=1,
                )
            self.assertEqual(r2["document"]["persistence"]["failed_entities"], 1)
            self.assertEqual(r2["document"]["stats"]["wall_seconds"], 1.5)
        finally:
            store.close()

    def test_describe_ignores_malformed_bool_or_negative_prev_wall_seconds(self):
        models1 = FakeModels(SERVICES, self._verify_and_persist_handler)
        store = Store(self.db_path)
        try:
            r1 = run_describe(
                store, self.run_id, services=SERVICES, models=models1,
                top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                persistence=0,
            )
            bad_states = [
                {"status": "completed", "views": 1, "verify": 0, "stats": {"wall_seconds": True}},
                {"status": "completed", "views": 1, "verify": 0, "stats": {"wall_seconds": -3.0}},
                {"status": "completed", "views": 1, "verify": 0, "stats": {"wall_seconds": "bad"}},
                {"status": "failed", "views": 1, "verify": 0, "stats": {"wall_seconds": 99.0}},
            ]
            for idx, bad_state in enumerate(bad_states, start=1):
                store.execute(
                    "UPDATE describe_runs SET state=? WHERE id=?",
                    (dumps(bad_state), r1["describe_id"]),
                )
                expected_elapsed = round(0.1 * idx, 4)
                with patch("entitytrace.describe.time.monotonic", side_effect=_monotonic_pairs(expected_elapsed)):
                    r_next = run_describe(
                        store, self.run_id, services=SERVICES, models=_NoCallModels(SERVICES),
                        top_k=10, kinds="person", concurrency=1, views=1, verify=0,
                        persistence=0,
                    )
                self.assertEqual(r_next["document"]["stats"]["wall_seconds"], expected_elapsed)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
