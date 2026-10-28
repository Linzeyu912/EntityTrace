from __future__ import annotations

import hashlib
from pathlib import Path

from .storage import Store
from .util import digest, dumps

OFFSET_UNIT = "unicode_code_point"


def import_text(store: Store, path: Path, encoding: str = "utf-8") -> dict:
    raw = path.read_bytes()
    # 只移除 UTF-8 BOM，保留 CRLF、空格和所有正文；原始字节另存。
    bom = encoding.lower().replace("_", "-") in {"utf-8", "utf8"} and raw.startswith(b"\xef\xbb\xbf")
    text = raw[3:].decode(encoding, errors="strict") if bom else raw.decode(encoding, errors="strict")
    if not text.strip():
        raise ValueError("输入文本为空")
    raw_hash = hashlib.sha256(raw).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source_id = digest(["text-v1", raw_hash, encoding, bom, text_hash])
    metadata = {"id": source_id, "filename": path.name, "raw_sha256": raw_hash,
                "text_sha256": text_hash, "encoding": encoding, "removed_utf8_bom": bom,
                "offset_unit": OFFSET_UNIT, "code_points": len(text), "raw_bytes": len(raw),
                "newline_policy": "preserved", "normalization": "none"}
    store.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?,?)", (source_id, dumps(metadata), text, raw))
    return store.source(source_id)


def address(source_id: str, start: int, end: int) -> dict:
    return {"text_version_id": source_id, "offset_unit": OFFSET_UNIT, "start": start, "end": end}


def validate_span(text: str, span: dict, source_id: str, quote: str | None = None) -> str:
    if span.get("text_version_id") != source_id or span.get("offset_unit") != OFFSET_UNIT:
        raise ValueError("地址的原文版本或计数单位不匹配")
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("地址范围无效")
    value = text[start:end]
    if quote is not None and value != quote:
        raise ValueError("地址取回的原文与摘录不一致")
    return value


def units(text: str, start: int = 0, end: int | None = None, limit: int = 700) -> list[dict]:
    """按原始物理行和安全长度产生定位单元，不改写原文。"""
    end = len(text) if end is None else end
    output = []
    cursor = start
    for line in text[start:end].splitlines(keepends=True):
        for offset in range(0, len(line), limit):
            a, b = cursor + offset, min(cursor + offset + limit, cursor + len(line))
            if text[a:b].strip():
                output.append({"id": f"u{a}_{b}", "start": a, "end": b, "text": text[a:b],
                               "complete_line": offset == 0 and b == cursor + len(line)})
        cursor += len(line)
    return output


def windows(items: list[dict], budget: int = 9000, count: int = 24) -> list[list[dict]]:
    batches: list[list[dict]] = []
    batch: list[dict] = []
    size = 0
    for item in items:
        n = len(item["text"])
        if batch and (size + n > budget or len(batch) >= count):
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += n
    if batch:
        batches.append(batch)
    return batches


def exact_matches(text: str, needle: str, start: int, end: int) -> list[tuple[int, int]]:
    if not needle:
        return []
    found = []
    cursor = start
    while True:
        at = text.find(needle, cursor, end)
        if at < 0:
            return found
        found.append((at, at + len(needle)))
        cursor = at + max(1, len(needle))
