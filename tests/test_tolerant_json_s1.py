from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from entitytrace.config import Service
from entitytrace.describe import check_output, ground_output
from entitytrace.models import ModelError, Models, parse_tolerant_json
from entitytrace.storage import Store
from entitytrace.util import dumps


def _openai_body(content: str, finish_reason: str = "stop") -> io.BytesIO:
    raw = {
        "model": "test-model",
        "choices": [{"finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"total_tokens": 10},
    }
    return io.BytesIO(dumps(raw).encode("utf-8"))


class _Scripted:
    """按顺序返回脚本的 fake opener，记录收到请求的次数。"""

    def __init__(self, responses: list[io.BytesIO]):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request, timeout=None):
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return item


def _make_models(opener, provider: str = "deepseek") -> tuple[Models, Store, Path]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    db_path = Path(tmp.name) / "test.sqlite3"
    store = Store(db_path)
    services = {
        "discovery": Service(name="discovery", url="https://example.com/v1",
                             model="test-model", key="k", provider=provider),
        "decision": Service(name="decision", url="https://example.com/v1",
                            model="test-model", key="k", provider=provider),
    }
    models = Models(store, "run-1", "inv", services, opener=opener, attempts=1)
    _OPEN_STORES.append((store, tmp))
    return models, store, db_path


_OPEN_STORES: list[tuple[Store, tempfile.TemporaryDirectory]] = []


def _call_details(store: Store) -> list[dict]:
    rows = store.rows("SELECT detail FROM calls ORDER BY created", ())
    return [json.loads(row["detail"]) for row in rows]


def _noop_validate(value):
    return None


class _StoreClosingTestCase(unittest.TestCase):
    def tearDown(self):
        while _OPEN_STORES:
            store, tmp = _OPEN_STORES.pop()
            store.close()
            tmp.cleanup()


class ParseTolerantJsonTests(_StoreClosingTestCase):
    def test_first_object_then_second_object_merged(self):
        out = parse_tolerant_json('{"a": 1}\n{"links": []}')
        self.assertEqual(out, {"a": 1, "links": []})

    def test_first_object_then_comma_fragment_merged(self):
        content = '{"mentions": [{"x": 1}]}\n,"links": [{"left": "甲", "right": "乙", "ref": "u1"}]}'
        out = parse_tolerant_json(content)
        self.assertEqual(out["mentions"], [{"x": 1}])
        self.assertEqual(out["links"], [{"left": "甲", "right": "乙", "ref": "u1"}])

    def test_list_fields_concatenated_and_existing_scalar_not_overwritten(self):
        content = '{"mentions": [{"x": 1}], "name": "首"}\n{"mentions": [{"x": 2}], "name": "次"}'
        out = parse_tolerant_json(content)
        self.assertEqual(out["mentions"], [{"x": 1}, {"x": 2}])
        self.assertEqual(out["name"], "首")

    def test_non_dict_with_extra_data_fails(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_tolerant_json("[1, 2] 3")

    def test_non_dict_extra_fragment_fails(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_tolerant_json('{"a": 1}\n[2, 3]')

    def test_strip_fence(self):
        out = parse_tolerant_json('```json\n{"a": 1}\n```', strip_fence=True)
        self.assertEqual(out, {"a": 1})


class CheckOutputToleranceTests(_StoreClosingTestCase):
    def test_missing_events_filled(self):
        value = {"facts": [{"entity_id": "e1", "facet": "age", "value": "青年",
                            "quote": "q", "source": "narration"}]}
        check_output(value)
        self.assertEqual(value["events"], [])

    def test_missing_facts_filled(self):
        value = {"events": [{"entity_id": "e1", "event_type": "upgrade",
                             "splits_entity": False, "value": "v", "quote": "q"}]}
        check_output(value)
        self.assertEqual(value["facts"], [])

    def test_both_missing_still_fails(self):
        with self.assertRaises(ModelError):
            check_output({"other": []})

    def test_non_dict_still_fails(self):
        with self.assertRaises(ModelError):
            check_output([1, 2])

    def test_non_list_facts_still_fails(self):
        with self.assertRaises(ModelError):
            check_output({"facts": "x", "events": []})

    def test_ground_output_missing_arrays_no_keyerror(self):
        task = {"id": "t1", "stage": "describe-d1", "text": "林舟出场。",
                "segments": [{"start": 0, "end": 5}], "start": 0,
                "entity_kinds": {"e1": "person"}, "mentions_by_entity": {},
                "default_entity_id": None, "view": 1}
        result = ground_output("d1", task, {"facts": []}, [], set())
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["events"], [])


class GenerateTolerantJsonTests(_StoreClosingTestCase):
    def test_fragment_merged_and_returned_dict_filled(self):
        content = '{"facts": [{"entity_id": "e1", "facet": "age", "value": "v", "quote": "q", "source": "narration"}]}\n{"events": []}'
        opener = _Scripted([_openai_body(content)])
        models, store, _ = _make_models(opener)
        parsed, _ = models.generate("sys", {"x": 1}, "stage", check_output)
        self.assertEqual(parsed["events"], [])
        self.assertEqual(len(parsed["facts"]), 1)
        self.assertEqual(opener.calls, 1)

    def test_truncated_output_still_fails_and_logged(self):
        opener = _Scripted([_openai_body('{"facts": []', finish_reason="length")])
        models, store, _ = _make_models(opener)
        with self.assertRaises(ModelError):
            models.generate("sys", {"x": 1}, "stage", check_output)
        details = _call_details(store)
        self.assertTrue(details and details[-1].get("validation_error"))
        self.assertIn("截断", details[-1]["validation_error"])

    def test_merged_output_still_invalid_fails(self):
        content = '{"a": 1}\n{"b": 2}'
        opener = _Scripted([_openai_body(content)])
        models, store, _ = _make_models(opener)
        with self.assertRaises(ModelError):
            models.generate("sys", {"x": 1}, "stage", check_output)
        self.assertTrue(_call_details(store))

    def test_content_head_tail_written_on_validation_failure(self):
        long_content = '{"facts": [{"v": "' + "测" * 300 + '"}]}\n{"events": []}'
        # 首对象里 facts 不是合法条目类型（dict 但 check_output 只要求 dict，改为触发非法：
        # 让条目不是 dict 触发失败）
        long_content = '{"facts": ["' + "测" * 300 + '"]}\n{"events": []}'
        opener = _Scripted([_openai_body(long_content)])
        models, store, _ = _make_models(opener)
        with self.assertRaises(ModelError):
            models.generate("sys", {"x": 1}, "stage", check_output)
        detail = _call_details(store)[-1]
        self.assertIn("content_head", detail)
        self.assertIn("content_tail", detail)
        self.assertEqual(detail["content_head"], long_content[:200])
        self.assertEqual(detail["content_tail"], long_content[-200:])
        self.assertLessEqual(len(detail["content_head"]), 200)
        self.assertLessEqual(len(detail["content_tail"]), 200)
        # request_cache 不写入 content_head/content_tail
        cache_rows = store.rows("SELECT response FROM request_cache", ())
        for row in cache_rows:
            self.assertNotIn("content_head", row["response"])
            self.assertNotIn("content_tail", row["response"])


if __name__ == "__main__":
    unittest.main()
