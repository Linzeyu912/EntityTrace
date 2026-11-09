"""校验《封神演义》描述事实与状态变化事件金标准文件（descriptions.json）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entitytrace.chapters import (
    build_chapters,
    collect_candidates,
    filter_headings_with_body,
    parse_chapter_number,
)
from entitytrace.util import digest

FACET_ENUMS: dict[str, set[str]] = {
    "person": {"age", "gender", "build", "face", "hair", "feature", "costume", "demeanor", "voice"},
    "place": {"terrain", "architecture", "scale", "atmosphere"},
    "object": {"shape", "size", "color_material", "visual_effect", "holder"},
    "ability": {"visual_effect", "sound_effect"},
}

ENTITY_TYPES = set(FACET_ENUMS.keys())
SOURCE_ENUMS = {"narration", "dialogue", "poem"}
EVENT_TYPE_ENUMS = {
    "costume_change",
    "injury",
    "disguise",
    "transform",
    "age_jump",
    "upgrade",
    "damage",
    "owner_change",
    "death",
    "other",
}
EXPECTED_SAMPLED_ORDINALS = [8, 24, 40, 56, 72, 88]
MAX_QUOTE_CHARS = 120


def _build_deterministic_chapters(raw_bytes: bytes) -> tuple[str, str, dict[int, dict]]:
    bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    text = raw_bytes[3:].decode("utf-8", errors="strict") if bom else raw_bytes.decode("utf-8", errors="strict")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source_id = digest(["text-v1", raw_hash, "utf-8", bom, text_hash])
    cands = collect_candidates(text)
    labeled = [{**c, "kind": "chapter"} for c in cands if parse_chapter_number(c["title"]) is not None]
    adopted, _ = filter_headings_with_body(text, labeled)
    _, chapters = build_chapters({"id": source_id, "text": text}, adopted)
    by_ord = {c["ordinal"]: c for c in chapters}
    return raw_hash, text, by_ord


def check_description_gold(
    descriptions_path: Path,
    roster_path: Path | None = None,
    mentions_path: Path | None = None,
) -> tuple[list[str], dict]:
    errors: list[str] = []
    if not descriptions_path.is_file():
        return [f"找不到金标准文件: {descriptions_path}"], {}

    try:
        data = json.loads(descriptions_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"JSON 解析失败: {exc}"], {}

    if not isinstance(data, dict):
        return ["顶层结构必须为 JSON object"], {}

    ref_dir = descriptions_path.parent
    if roster_path is None:
        roster_path = ref_dir / "roster.json"
    if mentions_path is None:
        mentions_path = ref_dir / "mentions.json"

    if not roster_path.is_file():
        errors.append(f"找不到 roster.json: {roster_path}")
        roster_by_id: dict[str, dict] = {}
    else:
        roster_obj = json.loads(roster_path.read_text(encoding="utf-8"))
        roster_by_id = {
            e["gold_id"]: e
            for e in roster_obj.get("entities", [])
            if isinstance(e, dict) and isinstance(e.get("gold_id"), str)
        }

    for req_str in ("book", "source_path", "source_sha256", "guideline", "labeler", "human_review"):
        val = data.get(req_str)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"缺少顶层非空字符串字段: {req_str}")

    if data.get("guideline") != "0.16a-D-v1":
        errors.append(f"guideline 必须为 '0.16a-D-v1'（实际 {data.get('guideline')!r}）")
    if data.get("labeler") != "agent:annotator":
        errors.append(f"labeler 必须为 'agent:annotator'（实际 {data.get('labeler')!r}）")
    if data.get("human_review") != "pending":
        errors.append(f"human_review 必须为 'pending'（实际 {data.get('human_review')!r}）")

    source_rel = data.get("source_path")
    novel_path = ROOT / source_rel if isinstance(source_rel, str) and source_rel else ROOT / "test/text-library/texts/07_中文长篇_封神演义_100回.txt"
    if not novel_path.is_file():
        errors.append(f"找不到原文文件: {novel_path}")
        return errors, {}

    raw_hash, text, ch_by_ord = _build_deterministic_chapters(novel_path.read_bytes())
    if data.get("source_sha256") != raw_hash:
        errors.append(f"source_sha256 不匹配: 期望 {raw_hash}，实际 {data.get('source_sha256')!r}")

    sampled = data.get("sampled_chapters")
    if not isinstance(sampled, list) or [c.get("ordinal") for c in sampled if isinstance(c, dict)] != EXPECTED_SAMPLED_ORDINALS:
        errors.append(f"sampled_chapters 回目序号必须为 {EXPECTED_SAMPLED_ORDINALS}")
    sampled_by_ord: dict[int, dict] = {}
    if isinstance(sampled, list):
        for ch_entry in sampled:
            if not isinstance(ch_entry, dict):
                errors.append("sampled_chapters 元素必须为 object")
                continue
            ord_no = ch_entry.get("ordinal")
            if ord_no not in ch_by_ord:
                errors.append(f"sampled_chapters 含未知回目 {ord_no!r}")
                continue
            det = ch_by_ord[ord_no]
            for k in ("title", "start", "body_start", "end"):
                if ch_entry.get(k) != det.get(k):
                    errors.append(f"第 {ord_no} 回 {k} 与确定性切章不一致: {ch_entry.get(k)!r} != {det.get(k)!r}")
            sampled_by_ord[ord_no] = ch_entry

    if mentions_path.is_file():
        mentions_obj = json.loads(mentions_path.read_text(encoding="utf-8"))
        if sampled != mentions_obj.get("sampled_chapters"):
            errors.append("sampled_chapters 与 mentions.json 不一致")

    facts = data.get("facts")
    events = data.get("events")
    skips = data.get("skips")
    if not isinstance(facts, list) or not facts:
        errors.append("facts 必须为非空数组")
        facts = []
    if not isinstance(events, list) or not events:
        errors.append("events 必须为非空数组")
        events = []
    if not isinstance(skips, list):
        errors.append("skips 必须为数组")
        skips = []

    seen_ids: set[str] = set()

    def _check_common_span_and_entity(item: dict, id_key: str, idx: int) -> None:
        item_id = item.get(id_key)
        tag = f"{id_key}={item_id!r}(#{idx})"
        if not isinstance(item_id, str) or not item_id.strip():
            errors.append(f"{tag}: 缺少非空字符串 {id_key}")
        elif item_id in seen_ids:
            errors.append(f"{tag}: ID 重复 ({item_id})")
        else:
            seen_ids.add(item_id)

        ch_ord = item.get("chapter_ordinal")
        if type(ch_ord) is not int or ch_ord not in sampled_by_ord:
            errors.append(f"{tag}: chapter_ordinal 必须属于抽样六回 {EXPECTED_SAMPLED_ORDINALS}（实际 {ch_ord!r}）")
            return

        start = item.get("start")
        end = item.get("end")
        quote = item.get("quote")
        if type(start) is not int or type(end) is not int:
            errors.append(f"{tag}: start/end 必须为整数（实际 {start!r}, {end!r}）")
            return
        if not isinstance(quote, str) or not quote:
            errors.append(f"{tag}: quote 必须为非空字符串")
            return
        if len(quote) > MAX_QUOTE_CHARS:
            errors.append(f"{tag}: quote 长度超过 {MAX_QUOTE_CHARS} 码点（实际 {len(quote)}）")
        if not (0 <= start < end <= len(text)):
            errors.append(f"{tag}: 区间 [{start}, {end}) 越界")
            return
        actual = text[start:end]
        if actual != quote:
            errors.append(f"{tag}: quote 与原文切片不一致: text[{start}:{end}]={actual!r} != {quote!r}")
        ch_info = sampled_by_ord[ch_ord]
        bs, be = ch_info["body_start"], ch_info["end"]
        if not (bs <= start < end <= be):
            errors.append(f"{tag}: 区间 [{start}, {end}) 未落在第 {ch_ord} 回正文区间 [{bs}, {be}) 内")

        surf = item.get("entity_surface")
        if not isinstance(surf, str) or not surf.strip():
            errors.append(f"{tag}: entity_surface 必须为非空字符串")

        etype = item.get("entity_type")
        if etype not in ENTITY_TYPES:
            errors.append(f"{tag}: entity_type 非法 ({etype!r})，允许值: {sorted(ENTITY_TYPES)}")

        gid = item.get("gold_id")
        if gid is not None:
            if not isinstance(gid, str) or gid not in roster_by_id:
                errors.append(f"{tag}: gold_id {gid!r} 不存在于 roster.json")
            else:
                r_ent = roster_by_id[gid]
                allowed_kinds = {r_ent.get("kind"), *(r_ent.get("alt_kinds") or [])}
                if etype not in allowed_kinds:
                    errors.append(
                        f"{tag}: gold_id {gid} 类别不匹配: entity_type={etype!r}, roster kind/alt_kinds={sorted(k for k in allowed_kinds if k)}"
                    )

        note = item.get("note")
        if not isinstance(note, str) or not note.strip():
            errors.append(f"{tag}: note 必须为非空字符串")

    for idx, f in enumerate(facts, 1):
        if not isinstance(f, dict):
            errors.append(f"facts[#{idx}] 必须为 object")
            continue
        _check_common_span_and_entity(f, "fact_id", idx)
        tag = f"fact_id={f.get('fact_id')!r}(#{idx})"
        etype = f.get("entity_type")
        facet = f.get("facet")
        if etype in FACET_ENUMS and facet not in FACET_ENUMS[etype]:
            errors.append(f"{tag}: facet {facet!r} 与类别 {etype!r} 不符，允许值: {sorted(FACET_ENUMS[etype])}")
        val = f.get("value")
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{tag}: value 必须为非空字符串")
        src = f.get("source")
        if src not in SOURCE_ENUMS:
            errors.append(f"{tag}: source {src!r} 非法，允许值: {sorted(SOURCE_ENUMS)}")

    for idx, ev in enumerate(events, 1):
        if not isinstance(ev, dict):
            errors.append(f"events[#{idx}] 必须为 object")
            continue
        _check_common_span_and_entity(ev, "event_id", idx)
        tag = f"event_id={ev.get('event_id')!r}(#{idx})"
        evtype = ev.get("event_type")
        if evtype not in EVENT_TYPE_ENUMS:
            errors.append(f"{tag}: event_type {evtype!r} 非法，允许值: {sorted(EVENT_TYPE_ENUMS)}")
        if type(ev.get("splits_entity")) is not bool:
            errors.append(f"{tag}: splits_entity 必须为布尔值（实际 {ev.get('splits_entity')!r}）")
        val = ev.get("value")
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{tag}: value 必须为非空字符串")

    for idx, sk in enumerate(skips, 1):
        if not isinstance(sk, dict):
            errors.append(f"skips[#{idx}] 必须为 object")
            continue
        _check_common_span_and_entity(sk, "skip_id", idx)
        tag = f"skip_id={sk.get('skip_id')!r}(#{idx})"
        reason = sk.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{tag}: reason 必须为非空字符串")

    summary = {
        "ok": len(errors) == 0,
        "path": str(descriptions_path),
        "sha256": hashlib.sha256(descriptions_path.read_bytes()).hexdigest(),
        "counts": {
            "facts": len(facts),
            "events": len(events),
            "skips": len(skips),
        },
        "errors": errors,
    }
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验描述事实与状态变化金标准文件（descriptions.json）。")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(ROOT / "test" / "entity-gold" / "fengshen" / "reference" / "descriptions.json"),
        help="descriptions.json 路径",
    )
    args = parser.parse_args(argv)
    errors, summary = check_description_gold(Path(args.path))
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
