from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entitytrace.chapters import BATCH_MAX_CANDIDATES, detect_chapters
from entitytrace.storage import Store
from entitytrace.text import import_text


class _RecordingModels:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def generate(self, system, state, stage, validate, max_tokens=8000, temperature=None):
        cands = state["candidates"]
        self.batch_sizes.append(len(cands))
        headings = [
            {"id": c["id"], "kind": "chapter"}
            for c in cands
            if str(c["text"]).startswith("第") and "章" in str(c["text"])
        ]
        val = {"headings": headings}
        validate(val)
        return val, f"req_{len(self.batch_sizes)}"


class ChaptersBatchSizeTests(unittest.TestCase):
    def test_batch_max_candidates_is_400_and_boundary_splitting(self):
        self.assertEqual(BATCH_MAX_CANDIDATES, 400)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = Store(root / "db.sqlite3")
            try:
                lines_400 = [f"第{i}章 标题{i}" if i <= 3 else f"短行内容_{i}" for i in range(1, 401)]
                txt_400 = root / "n400.txt"
                txt_400.write_text("\n".join(lines_400) + "\n", encoding="utf-8")
                src_400 = import_text(store, txt_400, "utf-8")
                store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("run_400", src_400["id"], "{}", "{}", "now", "now"))
                m_400 = _RecordingModels()
                out_400 = detect_chapters(store, m_400, src_400, "run_400", lambda _: None, concurrency=1)
                self.assertTrue(out_400["complete"])
                self.assertEqual(m_400.batch_sizes, [400])

                lines_401 = [f"第{i}章 标题{i}" if i <= 3 else f"短行内容_{i}" for i in range(1, 402)]
                txt_401 = root / "n401.txt"
                txt_401.write_text("\n".join(lines_401) + "\n", encoding="utf-8")
                src_401 = import_text(store, txt_401, "utf-8")
                store.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", ("run_401", src_401["id"], "{}", "{}", "now", "now"))
                m_401 = _RecordingModels()
                out_401 = detect_chapters(store, m_401, src_401, "run_401", lambda _: None, concurrency=1)
                self.assertTrue(out_401["complete"])
                self.assertEqual(m_401.batch_sizes, [400, 1])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
