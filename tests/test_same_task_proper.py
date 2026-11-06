from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.discovery import ground_result, plan_tasks
from entitytrace.pipeline import build_export_bundle
from entitytrace.propagation import deduplicate_grounding_reviews, propagate_same_task_proper
from entitytrace.roster import VERSION as ROSTER_VERSION, normalize_surface, resolve_roster
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


class SameTaskProperPropagationTests(unittest.TestCase):
    def _make_base_env(self, text: str):
        source_id = "src_synthetic_001"
        source = {"id": source_id, "text": text}
        chapters = [
            {
                "id": "ch1",
                "ordinal": 1,
                "kind": "chapter",
                "title": "第一章",
                "start": 0,
                "body_start": 4,
                "end": len(text),
                "chapter_set_version": "v1",
            }
        ]
        dtasks = plan_tasks(source, chapters, task_chars=9000)
        return source, chapters, dtasks

    def test_only_c1_proper_name_completed_and_weak_norms_excluded(self):
        text = "第一章\n沈墨走进客栈。沈墨看向掌柜，掌柜点头。李青也在场，沈墨又唤李青。"
        source, chapters, dtasks = self._make_base_env(text)
        dt = dtasks[0]
        u0 = dt["units"][0]

        s1 = text.index("沈墨")
        z1 = text.index("掌柜")
        l1 = text.index("李青")

        mentions = [
            {
                "id": "m_shen_1",
                "text": "沈墨",
                "start": s1,
                "end": s1 + 2,
                "address": address(source["id"], s1, s1 + 2),
                "chapter_id": "ch1",
                "task_id": dt["id"],
                "provisional_kinds": ["person"],
                "observation_ids": ["obs1"],
                "evidence": [address(source["id"], u0["start"], u0["end"])],
                "entity_id": "ent_shen",
                "link_method": "chapter_local",
                "link_refs": ["req_c1"],
            },
            {
                "id": "m_zhang_1",
                "text": "掌柜",
                "start": z1,
                "end": z1 + 2,
                "address": address(source["id"], z1, z1 + 2),
                "chapter_id": "ch1",
                "task_id": dt["id"],
                "provisional_kinds": ["person"],
                "observation_ids": ["obs2"],
                "evidence": [address(source["id"], u0["start"], u0["end"])],
                "entity_id": "ent_zhang",
                "link_method": "chapter_local",
                "link_refs": ["req_c1"],
            },
            {
                "id": "m_li_1",
                "text": "李青",
                "start": l1,
                "end": l1 + 2,
                "address": address(source["id"], l1, l1 + 2),
                "chapter_id": "ch1",
                "task_id": dt["id"],
                "provisional_kinds": ["person"],
                "observation_ids": ["obs3"],
                "evidence": [address(source["id"], u0["start"], u0["end"])],
                "entity_id": "ent_li",
                "link_method": "chapter_local",
                "link_refs": ["req_c1"],
            },
        ]
        entities = [
            {
                "id": "ent_shen",
                "name": "沈墨",
                "names": ["沈墨"],
                "aliases": ["沈墨"],
                "mention_ids": ["m_shen_1"],
                "mention_count": 1,
                "start": s1,
                "first_chapter_id": "ch1",
                "kind": "person",
                "identity_status": "linked_by_roster",
                "link_methods": {"chapter_local": 1},
            },
            {
                "id": "ent_zhang",
                "name": "掌柜",
                "names": ["掌柜"],
                "aliases": [],
                "mention_ids": ["m_zhang_1"],
                "mention_count": 1,
                "start": z1,
                "first_chapter_id": "ch1",
                "kind": "person",
                "identity_status": "linked_by_roster",
                "link_methods": {"chapter_local": 1},
            },
            {
                "id": "ent_li",
                "name": "李青",
                "names": ["李青"],
                "aliases": ["李青"],
                "mention_ids": ["m_li_1"],
                "mention_count": 1,
                "start": l1,
                "first_chapter_id": "ch1",
                "kind": "person",
                "identity_status": "linked_by_roster",
                "link_methods": {"chapter_local": 1},
            },
        ]
        c1_results = {
            dt["id"]: {
                "task_id": dt["id"],
                "request_id": "req_c1",
                "declared_proper_norms": [normalize_surface("沈墨"), normalize_surface("李青")],
                "weak_norms": [normalize_surface("李青")],
                "entities": [],
            }
        }
        summary = {"link_methods": {"chapter_local": 3}}
        ents_out, ms_out, _, sum_out, added = propagate_same_task_proper(
            source, chapters, dtasks, c1_results, entities, mentions, [], summary
        )
        self.assertEqual(len(added), 2)
        self.assertTrue(all(m["text"] == "沈墨" for m in added))
        self.assertTrue(all(m["link_method"] == "same_task_proper" for m in added))
        self.assertTrue(all(m["entity_id"] == "ent_shen" for m in added))
        shen_ent = next(e for e in ents_out if e["id"] == "ent_shen")
        self.assertEqual(shen_ent["mention_count"], 3)
        self.assertEqual(shen_ent["link_methods"]["same_task_proper"], 2)
        self.assertEqual(sum_out["link_methods"]["same_task_proper"], 2)

    def test_skip_when_surface_maps_to_multiple_entities_or_unresolved(self):
        text = "第一章\n王五与另一位王五交谈，随后王五离开。陈六看了看陈六。"
        source, chapters, dtasks = self._make_base_env(text)
        dt = dtasks[0]
        u0 = dt["units"][0]
        w1 = text.index("王五")
        w2 = text.index("王五", w1 + 2)
        c1 = text.index("陈六")

        mentions = [
            {
                "id": "m_w1",
                "text": "王五",
                "start": w1,
                "end": w1 + 2,
                "address": address(source["id"], w1, w1 + 2),
                "chapter_id": "ch1",
                "task_id": dt["id"],
                "provisional_kinds": ["person"],
                "observation_ids": [],
                "evidence": [address(source["id"], u0["start"], u0["end"])],
                "entity_id": "ent_w_a",
                "link_method": "chapter_local",
                "link_refs": [],
            },
            {
                "id": "m_w2",
                "text": "王五",
                "start": w2,
                "end": w2 + 2,
                "address": address(source["id"], w2, w2 + 2),
                "chapter_id": "ch1",
                "task_id": dt["id"],
                "provisional_kinds": ["person"],
                "observation_ids": [],
                "evidence": [address(source["id"], u0["start"], u0["end"])],
                "entity_id": "ent_w_b",
                "link_method": "chapter_local",
                "link_refs": [],
            },
            {
                "id": "m_c1",
                "text": "陈六",
                "start": c1,
                "end": c1 + 2,
                "address": address(source["id"], c1, c1 + 2),
                "chapter_id": "ch1",
                "task_id": dt["id"],
                "provisional_kinds": ["person"],
                "observation_ids": [],
                "evidence": [address(source["id"], u0["start"], u0["end"])],
                "entity_id": "ent_c",
                "link_method": "unresolved",
                "link_refs": [],
            },
        ]
        entities = [
            {
                "id": "ent_w_a",
                "name": "王五",
                "names": ["王五"],
                "aliases": ["王五"],
                "mention_ids": ["m_w1"],
                "mention_count": 1,
                "start": w1,
                "first_chapter_id": "ch1",
                "kind": "person",
                "identity_status": "linked_by_roster",
                "link_methods": {"chapter_local": 1},
            },
            {
                "id": "ent_w_b",
                "name": "王五",
                "names": ["王五"],
                "aliases": ["王五"],
                "mention_ids": ["m_w2"],
                "mention_count": 1,
                "start": w2,
                "first_chapter_id": "ch1",
                "kind": "person",
                "identity_status": "linked_by_roster",
                "link_methods": {"chapter_local": 1},
            },
            {
                "id": "ent_c",
                "name": "陈六",
                "names": ["陈六"],
                "aliases": [],
                "mention_ids": ["m_c1"],
                "mention_count": 1,
                "start": c1,
                "first_chapter_id": "ch1",
                "kind": "person",
                "identity_status": "unresolved_singleton",
                "link_methods": {"unresolved": 1},
            },
        ]
        c1_results = {
            dt["id"]: {
                "task_id": dt["id"],
                "request_id": "req_c1",
                "declared_proper_norms": [normalize_surface("王五"), normalize_surface("陈六")],
                "weak_norms": [],
                "entities": [],
            }
        }
        _, _, _, _, added = propagate_same_task_proper(
            source, chapters, dtasks, c1_results, entities, mentions, [], {}
        )
        self.assertEqual(added, [])

    def test_overlap_with_existing_mentions_and_longest_first_tie_breaking(self):
        text = "第一章\n青云剑与青云齐名；此处再提青云剑，又见甲乙与乙丙合称甲乙丙。"
        source, chapters, dtasks = self._make_base_env(text)
        dt = dtasks[0]
        u0 = dt["units"][0]

        qyj_1 = text.index("青云剑")
        qy_1 = text.index("青云", qyj_1 + 3)
        jy_1 = text.index("甲乙")
        yb_1 = text.index("乙丙")

        mentions = []
        entities = []
        for idx, (surf, pos, eid) in enumerate(
            [("青云剑", qyj_1, "e_qyj"), ("青云", qy_1, "e_qy"), ("甲乙", jy_1, "e_jy"), ("乙丙", yb_1, "e_yb")]
        ):
            mid = f"m_{idx}"
            mentions.append(
                {
                    "id": mid,
                    "text": surf,
                    "start": pos,
                    "end": pos + len(surf),
                    "address": address(source["id"], pos, pos + len(surf)),
                    "chapter_id": "ch1",
                    "task_id": dt["id"],
                    "provisional_kinds": ["object"],
                    "observation_ids": [],
                    "evidence": [address(source["id"], u0["start"], u0["end"])],
                    "entity_id": eid,
                    "link_method": "chapter_local",
                    "link_refs": [],
                }
            )
            entities.append(
                {
                    "id": eid,
                    "name": surf,
                    "names": [surf],
                    "aliases": [surf],
                    "mention_ids": [mid],
                    "mention_count": 1,
                    "start": pos,
                    "first_chapter_id": "ch1",
                    "kind": "object",
                    "identity_status": "linked_by_roster",
                    "link_methods": {"chapter_local": 1},
                }
            )

        c1_results = {
            dt["id"]: {
                "task_id": dt["id"],
                "request_id": "req_c1",
                "declared_proper_norms": [
                    normalize_surface("青云剑"),
                    normalize_surface("青云"),
                    normalize_surface("甲乙"),
                    normalize_surface("乙丙"),
                ],
                "weak_norms": [],
                "entities": [],
            }
        }
        _, _, _, _, added = propagate_same_task_proper(
            source, chapters, dtasks, c1_results, entities, mentions, [], {}
        )
        # 第二处“青云剑”应按长者优先选“青云剑”（不选其子串“青云”）；“甲乙丙”中等长重叠按起点先后选“甲乙”
        self.assertEqual([m["text"] for m in added], ["青云剑", "甲乙"])

    def test_english_word_boundary_respected(self):
        text = "第一章\nAnn met Anne and Marianne, then Ann waved."
        source, chapters, dtasks = self._make_base_env(text)
        dt = dtasks[0]
        u0 = dt["units"][0]
        a1 = text.index("Ann")
        mentions = [
            {
                "id": "m_ann_1",
                "text": "Ann",
                "start": a1,
                "end": a1 + 3,
                "address": address(source["id"], a1, a1 + 3),
                "chapter_id": "ch1",
                "task_id": dt["id"],
                "provisional_kinds": ["person"],
                "observation_ids": [],
                "evidence": [address(source["id"], u0["start"], u0["end"])],
                "entity_id": "ent_ann",
                "link_method": "chapter_local",
                "link_refs": [],
            }
        ]
        entities = [
            {
                "id": "ent_ann",
                "name": "Ann",
                "names": ["Ann"],
                "aliases": ["Ann"],
                "mention_ids": ["m_ann_1"],
                "mention_count": 1,
                "start": a1,
                "first_chapter_id": "ch1",
                "kind": "person",
                "identity_status": "linked_by_roster",
                "link_methods": {"chapter_local": 1},
            }
        ]
        c1_results = {
            dt["id"]: {
                "task_id": dt["id"],
                "request_id": "req_c1",
                "declared_proper_norms": [normalize_surface("Ann")],
                "weak_norms": [],
                "entities": [],
            }
        }
        _, _, _, _, added = propagate_same_task_proper(
            source, chapters, dtasks, c1_results, entities, mentions, [], {}
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["text"], "Ann")
        self.assertEqual(added[0]["start"], text.rindex("Ann"))

    def test_grounding_review_deduplication_and_byte_identical_when_nothing_to_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "第一章\n沈墨在青石渡饮茶，沈墨笑看江面。\n"
            book = root / "book.txt"
            book.write_bytes(text.encode("utf-8"))
            store = Store(root / "test.sqlite3")
            try:
                source = import_text(store, book)
                ver, chapters = build_chapters(
                    source,
                    [{"start": 0, "end": 3, "title": "第一章", "kind": "chapter"}],
                )
                run_id = "run_prop_test_001"
                cfg = {"identity_algorithm": ROSTER_VERSION, "task_chars": 9000}
                state = {
                    "chapters": {"complete": True, "chapter_set_version": ver, "issues": 0, "path_counts": {"adopted": 1}},
                    "discovery": {"complete": True, "tasks": 1, "completed": 1},
                }
                store.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                    (run_id, source["id"], dumps(cfg), dumps(state), "now", "now"),
                )
                for ch in chapters:
                    store.put(run_id, "chapter", ch)
                dtasks = plan_tasks(source, chapters, 9000)
                dt = dtasks[0]
                u0 = dt["units"][0]["id"]
                # 只报第一处“沈墨”，并构造一条“沈墨”在不存在/空单元上的 not-found 待审与一条“幽冥谷”未覆盖待审
                s1 = text.index("沈墨")
                m1 = {
                    "id": "m_shen_1",
                    "text": "沈墨",
                    "start": s1,
                    "end": s1 + 2,
                    "address": address(source["id"], s1, s1 + 2),
                    "chapter_id": chapters[0]["id"],
                    "task_id": dt["id"],
                    "provisional_kinds": ["person"],
                    "observation_ids": ["obs_shen"],
                    "evidence": [address(source["id"], dt["units"][0]["start"], dt["units"][0]["end"])],
                }
                obs_shen = {
                    "id": "obs_shen",
                    "task_id": dt["id"],
                    "request_id": "req_d1",
                    "text": "沈墨",
                    "kind": "person",
                    "reported_refs": [u0],
                    "mention_ids": ["m_shen_1"],
                    "status": "grounded",
                }
                obs_unm = {
                    "id": "obs_unm",
                    "task_id": dt["id"],
                    "request_id": "req_d1",
                    "text": "幽冥谷",
                    "kind": "place",
                    "reported_refs": [u0],
                    "mention_ids": [],
                    "status": "unlocated",
                }
                rev_shen = {
                    "id": "rev_shen_nf",
                    "stage": "grounding",
                    "reason": "原文引用内找不到完整称呼",
                    "observation_id": "obs_shen",
                    "evidence": [address(source["id"], dt["units"][0]["start"], dt["units"][0]["end"])],
                    "status": "pending",
                }
                rev_unm = {
                    "id": "rev_unm_nf",
                    "stage": "grounding",
                    "reason": "原文引用内找不到完整称呼",
                    "observation_id": "obs_unm",
                    "evidence": [address(source["id"], dt["units"][0]["start"], dt["units"][0]["end"])],
                    "status": "pending",
                }
                store.finish_task(
                    run_id,
                    dt,
                    {"request_id": "req_d1"},
                    [
                        ("mention", m1),
                        ("observation", obs_shen),
                        ("observation", obs_unm),
                        ("review", rev_shen),
                        ("review", rev_unm),
                    ],
                )

                models = FakeRosterModels(
                    lambda st: {
                        "entities": [
                            {
                                "local_id": "e1",
                                "name": "沈墨",
                                "kind": "person",
                                "surfaces": [{"surface": "沈墨", "surface_type": "proper_name"}],
                                "evidence_ref": u0,
                            }
                        ],
                        "assignments": [],
                        "not_entity": [],
                    }
                )
                out = resolve_roster(store, models, source, run_id, concurrency=1, log=lambda *_: None, task_chars=9000)
                self.assertTrue(out["complete"])
                store.state(run_id, identity=out)

                bundle1 = build_export_bundle(store, run_id)
                # 第二处“沈墨”已被同任务补全
                self.assertEqual(len(bundle1["result"]["mentions"]), 2)
                self.assertEqual(bundle1["result"]["roster_summary"]["link_methods"].get("same_task_proper"), 1)
                # rev_shen_nf 自动消解为 auto_same_task_covered，而 rev_unm_nf 仍为 pending
                revs_by_id = {r["id"]: r for r in bundle1["review"]["items"]}
                self.assertEqual(revs_by_id["rev_shen_nf"]["status"], "resolved")
                self.assertEqual(revs_by_id["rev_shen_nf"]["resolution"], "auto_same_task_covered")
                self.assertEqual(revs_by_id["rev_unm_nf"]["status"], "pending")
                self.assertEqual(bundle1["review"]["counts"]["pending"], 1)
                self.assertEqual(bundle1["review"]["counts"]["resolved"], 1)
                self.assertEqual(bundle1["manifest"]["counts"]["review_items"], 1)

                # 再次对无可补的已完成结果执行 propagate_same_task_proper，验证逐字节不变
                roster_out = store.records(run_id, "roster_output")[0]
                e2, m2, a2, s2, added2 = propagate_same_task_proper(
                    source,
                    chapters,
                    dtasks,
                    {},
                    roster_out["entities"],
                    roster_out["mentions"],
                    roster_out["ambiguous_surfaces"],
                    roster_out["roster_summary"],
                )
                self.assertEqual(added2, [])
                self.assertEqual(dumps(e2), dumps(roster_out["entities"]))
                self.assertEqual(dumps(m2), dumps(roster_out["mentions"]))

                # 验证 build_export_bundle 对多出提及 link_method!='same_task_proper' 抛 ValueError
                tampered_out = dict(roster_out)
                tampered_mentions = [dict(x) for x in roster_out["mentions"]]
                for tm in tampered_mentions:
                    if tm["id"] != "m_shen_1":
                        tm["link_method"] = "book_proper"
                tampered_out["mentions"] = tampered_mentions
                store.put(run_id, "roster_output", tampered_out)
                with self.assertRaises(ValueError):
                    build_export_bundle(store, run_id)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
