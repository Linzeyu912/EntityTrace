from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import build_chapters
from entitytrace.config import Service
from entitytrace.discovery import PROMPT, check_discovery, discover, ground_result, plan_tasks
from entitytrace.models import ModelError, Models
from entitytrace.storage import Store
from entitytrace.text import import_text


class DiscoveryEmptyLinksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "db.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_prompt_sha256_and_task_id_unchanged(self):
        self.assertEqual(
            hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "a4e96c142f797c946f904a0fcba2b0be33db3b69351cfec1ce3f7cb36461c264",
        )
        src = {"id": "src_fixed", "text": "第一章\n林舟前行。\n"}
        _, chs = build_chapters(src, [{"start": 0, "end": 3, "title": "第一章", "kind": "chapter"}])
        tasks = plan_tasks(src, chs, 9000)
        self.assertEqual(
            tasks[0]["id"],
            "fc498302d584245d604686b4c0380b3803f1358f93c68039c3227f017fb89c3e",
        )

    def test_omitted_and_null_links_accepted_through_models_generate_and_ground_result(self):
        text = "第一章\n林舟走向渡口。\n第二章\n陈叔摇橹靠岸。\n"
        path = self.root / "novel.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        idx1, idx2 = text.index("第一章"), text.index("第二章")
        _, chapters = build_chapters(
            source,
            [
                {"start": idx1, "end": idx1 + 3, "title": "第一章", "kind": "chapter"},
                {"start": idx2, "end": idx2 + 3, "title": "第二章", "kind": "chapter"},
            ],
        )
        tasks = plan_tasks(source, chapters, 9000)
        u1 = tasks[0]["units"][0]["id"]
        u2 = tasks[1]["units"][0]["id"]

        # 第一章模拟完全省略顶层 "links" 键；第二章模拟返回 "links": null
        payloads_by_unit = {
            u1: json.dumps({"mentions": [{"text": "林舟", "kind": "person", "refs": [u1]}]}, ensure_ascii=False),
            u2: json.dumps({"mentions": [{"text": "陈叔", "kind": "person", "refs": [u2]}], "links": None}, ensure_ascii=False),
        }

        def opener(request, **kwargs):
            req_data = json.loads(request.data.decode("utf-8"))
            user_state = json.loads(req_data["messages"][1]["content"])
            uid = user_state["units"][0]["id"]
            content = payloads_by_unit[uid]
            raw = {"choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {}}
            return io.BytesIO(json.dumps(raw).encode("utf-8"))

        svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")

        # 第一次运行：走真实 HTTP 响应解析 -> Models.generate 二次 json.loads -> ground_result
        run1 = "run_cold"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run1, source["id"], "{}", "{}", "now", "now"))
        models1 = Models(self.store, run1, "inv1", {"discovery": svc}, attempts=1, opener=opener)
        out1 = discover(self.store, models1, source, run1, chapters, 9000, 1, lambda _: None)
        self.assertTrue(out1["complete"], out1)
        self.assertEqual(len(self.store.records(run1, "mention")), 2)
        self.assertEqual(self.store.records(run1, "link"), [])

        # 第二次新运行复用同一 request_cache：走缓存原文二次解析 -> ground_result，不触发 opener
        def fail_opener(request, **kwargs):
            raise AssertionError("缓存命中路径不应发起网络调用")

        run2 = "run_cached"
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run2, source["id"], "{}", "{}", "now", "now"))
        models2 = Models(self.store, run2, "inv2", {"discovery": svc}, attempts=1, opener=fail_opener)
        out2 = discover(self.store, models2, source, run2, chapters, 9000, 1, lambda _: None)
        self.assertTrue(out2["complete"], out2)
        self.assertEqual(len(self.store.records(run2, "mention")), 2)
        self.assertEqual(self.store.records(run2, "link"), [])

    def test_invalid_links_missing_mentions_and_bad_json_still_rejected(self):
        text = "第一章\n林舟前行。\n"
        path = self.root / "one.txt"
        path.write_bytes(text.encode("utf-8"))
        source = import_text(self.store, path)
        _, chapters = build_chapters(source, [{"start": 0, "end": 3, "title": "第一章", "kind": "chapter"}])
        task = plan_tasks(source, chapters, 9000)[0]

        for bad_value in [
            {"mentions": [], "links": "x"},
            {"mentions": [], "links": 0},
            {"mentions": [], "links": False},
            {"mentions": [], "links": {}},
            {"links": []},
            {"mentions": None, "links": []},
            {"mentions": "x"},
            {},
        ]:
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ModelError):
                    check_discovery(bad_value)
                with self.assertRaises(ModelError):
                    ground_result(source, task, dict(bad_value), "req_bad")

        # 无效 JSON 在 Models.generate 中仍抛 ModelError
        def bad_json_opener(request, **kwargs):
            raw = {"choices": [{"finish_reason": "stop", "message": {"content": "{not-valid-json"}}]}
            return io.BytesIO(json.dumps(raw).encode("utf-8"))

        svc = Service("discovery", "https://api.deepseek.com", "deepseek-flash", "TEST_SECRET")
        self.store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("run_bad", source["id"], "{}", "{}", "now", "now"))
        models = Models(self.store, "run_bad", "inv_bad", {"discovery": svc}, attempts=1, opener=bad_json_opener)
        with self.assertRaises(ModelError):
            models.generate(PROMPT, {"units": task["units"]}, "discovery", check_discovery)


if __name__ == "__main__":
    unittest.main()
