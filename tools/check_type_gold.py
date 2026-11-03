"""校验《封神演义》0.14a 多类型金标准三层文件（roster.json, mentions.json, hard_cases.json）。"""
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

ROSTER_ENTITY_TYPES = {"person", "place", "org", "item"}
MENTION_ENTITY_TYPES = {"person", "place", "org", "item", "generic", "not_entity"}
NON_PERSON_MENTION_TYPES = {"place", "org", "item", "generic", "not_entity"}
PROPER_ENTITY_TYPES = {"person", "place", "org", "item"}
NULL_GOLD_ID_TYPES = {"generic", "not_entity"}


def _build_deterministic_chapters(raw_bytes: bytes) -> tuple[str, str, dict[int, dict]]:
    bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    text = raw_bytes[3:].decode("utf-8", errors="strict") if bom else raw_bytes.decode("utf-8", errors="strict")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source_id = digest(["text-v1", raw_hash, "utf-8", bom, text_hash])
    cands = collect_candidates(text)
    labeled = [{**c, "kind": "chapter"} for c in cands if parse_chapter_number(c["title"]) is not None]
    adopted, _ = filter_headings_with_body(text, labeled)
    version, chapters = build_chapters({"id": source_id, "text": text}, adopted)
    by_ord = {c["ordinal"]: c for c in chapters}
    return raw_hash, version, by_ord


def _check_top_meta(obj: dict, fname: str, expected_sha256: str, errors: list[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"{fname}: 顶层结构必须为 JSON object")
        return
    for req_key in ("book", "source_path", "source_sha256", "guideline", "created_at", "labeler", "human_review"):
        val = obj.get(req_key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{fname}: 缺少顶层非空字符串字段 {req_key}")
    if obj.get("source_sha256") != expected_sha256:
        errors.append(
            f"{fname}: source_sha256 不匹配（期望 {expected_sha256}，实际 {obj.get('source_sha256')}）"
        )
    if obj.get("guideline") != "0.14a-T1-T5":
        errors.append(f"{fname}: guideline 必须为 '0.14a-T1-T5'（实际 {obj.get('guideline')!r}）")
    if obj.get("human_review") != "pending":
        errors.append(f"{fname}: human_review 必须为 'pending'（实际 {obj.get('human_review')!r}）")


def _check_item_provenance(item: dict, tag: str, errors: list[str]) -> None:
    note = item.get("note")
    if not isinstance(note, str) or not note.strip():
        errors.append(f"{tag}: 缺少非空 note 字段")
    if item.get("guideline") != "0.14a-T1-T5":
        errors.append(f"{tag}: guideline 必须为 '0.14a-T1-T5'")
    labeler = item.get("labeler")
    if not isinstance(labeler, str) or not labeler.startswith("agent:"):
        errors.append(f"{tag}: labeler 必须为 'agent:<model>' 格式")
    if item.get("human_review") != "pending":
        errors.append(f"{tag}: human_review 必须为 'pending'")


def _check_span_in_chapter(
    text: str,
    ch_by_ord: dict[int, dict],
    ch_ord: object,
    start: object,
    end: object,
    expected_surface: object,
    tag: str,
    errors: list[str],
) -> tuple[int, int] | None:
    if type(ch_ord) is not int or ch_ord not in ch_by_ord:
        errors.append(f"{tag}: chapter_ordinal 无效或不存在 ({ch_ord!r})")
        return None
    if type(start) is not int or type(end) is not int:
        errors.append(f"{tag}: start/end 必须为整数 ({start!r}, {end!r})")
        return None
    if not (0 <= start < end <= len(text)):
        errors.append(f"{tag}: 区间 [{start}, {end}) 越出全文范围 [0, {len(text)})")
        return None
    if not isinstance(expected_surface, str) or not expected_surface:
        errors.append(f"{tag}: text 必须为非空字符串")
        return None
    actual_surface = text[start:end]
    if actual_surface != expected_surface:
        errors.append(
            f"{tag}: 原文切片不一致，text[{start}:{end}]={actual_surface!r} != {expected_surface!r}"
        )
    ch = ch_by_ord[ch_ord]
    bs, be = ch["body_start"], ch["end"]
    if not (bs <= start < end <= be):
        errors.append(
            f"{tag}: 坐标 [{start}, {end}) 未落在第 {ch_ord} 回正文区间 [{bs}, {be}) 内"
        )
    return (start, end)


def _check_no_duplicate_or_overlapping_intervals(
    intervals: list[tuple[int, int, str]],
    layer_name: str,
    errors: list[str],
) -> None:
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    for i in range(len(sorted_iv) - 1):
        s1, e1, tag1 = sorted_iv[i]
        s2, e2, tag2 = sorted_iv[i + 1]
        if s1 == s2 and e1 == e2:
            errors.append(f"{layer_name}: 存在重复区间 [{s1}, {e1}) ({tag1} 与 {tag2})")
        elif s2 < e1:
            errors.append(
                f"{layer_name}: 存在重叠区间 [{s1}, {e1}) ({tag1}) 与 [{s2}, {e2}) ({tag2})"
            )


def _check_type_and_refs(
    item: dict,
    allowed_types: set[str],
    roster_ids: set[str],
    tag: str,
    errors: list[str],
) -> None:
    etype = item.get("entity_type")
    if etype not in allowed_types:
        errors.append(f"{tag}: 非法 entity_type {etype!r}（允许值：{sorted(allowed_types)}）")
        return
    gid = item.get("gold_id")
    ref_to = item.get("refers_to")
    if etype in PROPER_ENTITY_TYPES:
        if not isinstance(gid, str) or not gid:
            errors.append(f"{tag}: 专名类型 {etype!r} 的 gold_id 不能为空")
        elif gid not in roster_ids:
            errors.append(f"{tag}: gold_id {gid!r} 在 roster.json 中不存在")
    elif etype in NULL_GOLD_ID_TYPES:
        if gid is not None:
            errors.append(f"{tag}: 类型 {etype!r} 的 gold_id 必须为 null（实际为 {gid!r}）")
    if ref_to is not None:
        if not isinstance(ref_to, str) or ref_to not in roster_ids:
            errors.append(f"{tag}: refers_to {ref_to!r} 在 roster.json 中不存在")
        if etype != "generic":
            errors.append(f"{tag}: 仅 generic 类型可设置非空 refers_to（当前类型为 {etype!r}）")


def check_type_gold(gold_dir: Path, text_path: Path) -> dict:
    errors: list[str] = []
    if not text_path.is_file():
        return {"ok": False, "error_count": 1, "errors": [f"原文文件不存在: {text_path}"], "summary": {}}

    raw_bytes = text_path.read_bytes()
    bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    text = raw_bytes[3:].decode("utf-8", errors="strict") if bom else raw_bytes.decode("utf-8", errors="strict")
    expected_sha256, det_ch_ver, ch_by_ord = _build_deterministic_chapters(raw_bytes)

    roster_path = gold_dir / "roster.json"
    mentions_path = gold_dir / "mentions.json"
    hard_cases_path = gold_dir / "hard_cases.json"

    for p in (roster_path, mentions_path, hard_cases_path):
        if not p.is_file():
            errors.append(f"缺少金标准文件: {p.name}")
    if errors:
        return {"ok": False, "error_count": len(errors), "errors": errors, "summary": {}}

    try:
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        mentions = json.loads(mentions_path.read_text(encoding="utf-8"))
        hard_cases = json.loads(hard_cases_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error_count": 1, "errors": [f"JSON 解析失败: {exc}"], "summary": {}}

    _check_top_meta(roster, "roster.json", expected_sha256, errors)
    _check_top_meta(mentions, "mentions.json", expected_sha256, errors)
    _check_top_meta(hard_cases, "hard_cases.json", expected_sha256, errors)

    # --- 1. Check roster.json ---
    roster_entities = roster.get("entities")
    roster_ids: set[str] = set()
    roster_intervals: list[tuple[int, int, str]] = []
    roster_by_type: dict[str, int] = {}

    if not isinstance(roster_entities, list) or not roster_entities:
        errors.append("roster.json: entities 必须为非空数组")
    else:
        for idx, ent in enumerate(roster_entities):
            tag = f"roster.entities[{idx}]({ent.get('gold_id')})"
            gid = ent.get("gold_id")
            if not isinstance(gid, str) or not gid.strip():
                errors.append(f"{tag}: gold_id 必须为非空字符串")
            elif gid in roster_ids:
                errors.append(f"{tag}: 重复的 gold_id {gid!r}")
            else:
                roster_ids.add(gid)

            etype = ent.get("entity_type")
            if etype not in ROSTER_ENTITY_TYPES:
                errors.append(f"{tag}: 非法 entity_type {etype!r}")
            else:
                roster_by_type[etype] = roster_by_type.get(etype, 0) + 1

            cname = ent.get("canonical_name")
            if not isinstance(cname, str) or not cname.strip():
                errors.append(f"{tag}: canonical_name 必须为非空字符串")
            aliases = ent.get("aliases")
            if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) and a.strip() for a in aliases):
                errors.append(f"{tag}: aliases 必须为非空字符串列表")
            elif cname and cname not in aliases:
                errors.append(f"{tag}: canonical_name {cname!r} 必须包含在 aliases 中")

            _check_item_provenance(ent, tag, errors)

            rep_mentions = ent.get("representative_mentions")
            if not isinstance(rep_mentions, list) or not rep_mentions:
                errors.append(f"{tag}: representative_mentions 必须至少包含 1 条原文证据")
            else:
                for r_idx, rm in enumerate(rep_mentions):
                    r_tag = f"{tag}.rep[{r_idx}]"
                    iv = _check_span_in_chapter(
                        text, ch_by_ord, rm.get("chapter_ordinal"), rm.get("start"), rm.get("end"), rm.get("text"), r_tag, errors
                    )
                    if iv is not None:
                        roster_intervals.append((iv[0], iv[1], r_tag))

        _check_no_duplicate_or_overlapping_intervals(roster_intervals, "roster.json", errors)

    # --- 2. Check mentions.json ---
    sampled_chs = mentions.get("sampled_chapters")
    sampled_ords: set[int] = set()
    if not isinstance(sampled_chs, list) or not sampled_chs:
        errors.append("mentions.json: sampled_chapters 必须为非空数组")
    else:
        for sc in sampled_chs:
            ord_no = sc.get("ordinal")
            if type(ord_no) is not int or ord_no not in ch_by_ord:
                errors.append(f"mentions.sampled_chapters: 无效章节 ordinal {ord_no!r}")
                continue
            sampled_ords.add(ord_no)
            det_c = ch_by_ord[ord_no]
            if sc.get("body_start") != det_c["body_start"] or sc.get("end") != det_c["end"]:
                errors.append(
                    f"mentions.sampled_chapters(ch={ord_no}): [body_start, end) 与原文分章不一致"
                )

    m_list = mentions.get("mentions")
    s_list = mentions.get("skips", [])
    mention_intervals: list[tuple[int, int, str]] = []
    mentions_by_type: dict[str, int] = {}
    seen_mention_ids: set[str] = set()

    if not isinstance(m_list, list) or not m_list:
        errors.append("mentions.json: mentions 必须为非空数组")
    else:
        if len(m_list) > 1200:
            errors.append(f"mentions.json: mentions 数量 ({len(m_list)}) 超出 1,200 上限")
        for idx, m in enumerate(m_list):
            mid = m.get("mention_id")
            tag = f"mentions.mentions[{idx}]({mid})"
            if not isinstance(mid, str) or not mid:
                errors.append(f"{tag}: 缺少 mention_id")
            elif mid in seen_mention_ids:
                errors.append(f"{tag}: 重复的 mention_id {mid!r}")
            else:
                seen_mention_ids.add(mid)

            ch_ord = m.get("chapter_ordinal")
            if ch_ord not in sampled_ords:
                errors.append(f"{tag}: chapter_ordinal {ch_ord!r} 不在 sampled_chapters {sorted(sampled_ords)} 中")
            iv = _check_span_in_chapter(
                text, ch_by_ord, ch_ord, m.get("start"), m.get("end"), m.get("text"), tag, errors
            )
            if iv is not None:
                mention_intervals.append((iv[0], iv[1], tag))

            _check_type_and_refs(m, NON_PERSON_MENTION_TYPES, roster_ids, tag, errors)
            etype = m.get("entity_type")
            if isinstance(etype, str):
                mentions_by_type[etype] = mentions_by_type.get(etype, 0) + 1
            _check_item_provenance(m, tag, errors)

    if not isinstance(s_list, list):
        errors.append("mentions.json: skips 必须为数组")
    else:
        for idx, sk in enumerate(s_list):
            sid = sk.get("skip_id")
            tag = f"mentions.skips[{idx}]({sid})"
            ch_ord = sk.get("chapter_ordinal")
            if ch_ord not in sampled_ords:
                errors.append(f"{tag}: chapter_ordinal {ch_ord!r} 不在 sampled_chapters 中")
            iv = _check_span_in_chapter(
                text, ch_by_ord, ch_ord, sk.get("start"), sk.get("end"), sk.get("text"), tag, errors
            )
            if iv is not None:
                mention_intervals.append((iv[0], iv[1], tag))
            _check_type_and_refs(sk, {"not_entity"}, roster_ids, tag, errors)
            _check_item_provenance(sk, tag, errors)

    _check_no_duplicate_or_overlapping_intervals(mention_intervals, "mentions.json", errors)

    # --- 3. Check hard_cases.json ---
    cases = hard_cases.get("cases")
    hc_intervals: list[tuple[int, int, str]] = []
    seen_case_ids: set[str] = set()
    hc_mention_count = 0

    if not isinstance(cases, list) or not cases:
        errors.append("hard_cases.json: cases 必须为非空数组")
    else:
        for c_idx, case in enumerate(cases):
            cid = case.get("case_id")
            c_tag = f"hard_cases.cases[{c_idx}]({cid})"
            if not isinstance(cid, str) or not cid:
                errors.append(f"{c_tag}: 缺少 case_id")
            elif cid in seen_case_ids:
                errors.append(f"{c_tag}: 重复的 case_id {cid!r}")
            else:
                seen_case_ids.add(cid)

            _check_item_provenance(case, c_tag, errors)
            target_gids = case.get("target_gold_ids")
            if not isinstance(target_gids, list):
                errors.append(f"{c_tag}: target_gold_ids 必须为数组")
            else:
                for tg in target_gids:
                    if tg not in roster_ids:
                        errors.append(f"{c_tag}: target_gold_id {tg!r} 在 roster.json 中不存在")

            c_mentions = case.get("mentions")
            if not isinstance(c_mentions, list) or len(c_mentions) < 2:
                errors.append(f"{c_tag}: 每个难例必须包含至少 2 条 mentions")
            else:
                for m_idx, hm in enumerate(c_mentions):
                    hm_tag = f"{c_tag}.mentions[{m_idx}]({hm.get('mention_id')})"
                    iv = _check_span_in_chapter(
                        text, ch_by_ord, hm.get("chapter_ordinal"), hm.get("start"), hm.get("end"), hm.get("text"), hm_tag, errors
                    )
                    if iv is not None:
                        hc_intervals.append((iv[0], iv[1], hm_tag))
                    _check_type_and_refs(hm, MENTION_ENTITY_TYPES, roster_ids, hm_tag, errors)
                    _check_item_provenance(hm, hm_tag, errors)
                    hc_mention_count += 1

        _check_no_duplicate_or_overlapping_intervals(hc_intervals, "hard_cases.json", errors)

    summary = {
        "source_sha256": expected_sha256,
        "chapters_version": det_ch_ver,
        "chapter_count": len(ch_by_ord),
        "roster_total": len(roster_entities) if isinstance(roster_entities, list) else 0,
        "roster_by_type": roster_by_type,
        "sampled_chapters": sorted(sampled_ords),
        "mentions_total": len(m_list) if isinstance(m_list, list) else 0,
        "skips_total": len(s_list) if isinstance(s_list, list) else 0,
        "mentions_by_type": mentions_by_type,
        "hard_cases_total": len(cases) if isinstance(cases, list) else 0,
        "hard_case_mentions_total": hc_mention_count,
    }
    return {
        "ok": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 0.14a 多类型金标准三层文件")
    parser.add_argument("--gold", required=True, type=Path, help="金标准目录（含 roster.json, mentions.json, hard_cases.json）")
    parser.add_argument("--text", required=True, type=Path, help="对应小说原文路径")
    args = parser.parse_args(argv)

    report = check_type_gold(args.gold, args.text)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
