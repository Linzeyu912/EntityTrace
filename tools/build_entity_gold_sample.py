"""按 R3 规范从原文、金标准名册及共有发现提及生成分层抽样待标注清单（不含任何金标准答案）。"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entitytrace.discovery import _word_boundary
from entitytrace.roster import normalize_surface
from entitytrace.text import exact_matches
from entitytrace.util import write_json


EN_TITLE_WORDS = {
    "mr.", "mrs.", "miss", "ms.", "lady", "lord", "sir", "colonel", "captain",
    "general", "lieutenant", "doctor", "dr.", "inspector", "constable", "minister",
    "secretary", "premier", "grace", "ladyship", "madam", "ma'am", "gentleman", "officer",
}

EN_REL_PREFIXES = (
    "my ", "his ", "her ", "your ", "our ", "their ", "the ", "a ", "an ",
    "this ", "that ", "one of ", "two of ", "some of ",
)

EN_REL_WORDS = {
    "father", "mother", "sister", "sisters", "brother", "brothers", "aunt", "uncle",
    "cousin", "cousins", "niece", "nephew", "wife", "husband", "daughter", "daughters",
    "son", "sons", "child", "children", "friend", "friends", "master", "mistress",
    "servant", "servants", "housekeeper", "butler", "maid", "footman", "agent",
    "patroness", "owner", "neighbour", "neighbours", "companion", "partner", "family",
    "girls", "ladies", "gentlemen", "party", "regiment",
}

ZH_TITLE_SUBSTRS = (
    "大爷", "老爷", "太太", "夫人", "少爷", "小姐", "老爹", "老太", "大人", "太守",
    "知县", "总督", "制台", "制军", "抚院", "抚台", "中丞", "按察", "钦差", "县主",
    "乡绅", "生员", "举人", "局长", "巡警", "长官", "先生", "相公", "官人", "太爷",
    "藩台", "臬台", "道台", "知府", "通判", "教谕", "掌柜", "东家", "和尚", "道士",
)

ZH_REL_SUBSTRS = (
    "父", "母", "爹", "娘", "叔", "伯", "舅", "姑", "姨", "兄", "弟", "姐", "妹",
    "哥", "嫂", "侄", "甥", "妻", "妾", "媳", "婿", "婆", "公", "祖", "孙", "亲",
    "友", "伙计", "小厮", "家人", "丫鬟", "婢", "仆", "差役", "地保", "地方",
    "保正", "邻佑", "更夫", "师爷", "书办", "门公", "禁子", "狱卒", "稳婆", "媒婆",
    "省城", "县城", "府城", "会馆", "胡同",
)


def find_chapter_index(chapters: list[dict], start: int) -> int:
    for ch in chapters:
        sec_start = ch.get("section_start", ch.get("start", 0))
        sec_end = ch.get("body_end", ch.get("end", 0))
        if sec_start <= start < sec_end:
            return ch["index"]
    return 1


def deduplicate_longest_non_overlapping(matches: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    """在同一文本位置若存在包含关系（如 'Mr. Darcy' 与 'Darcy'），优先保留最长完整称呼。"""
    ordered = sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    accepted: list[tuple[int, int, str, str]] = []
    for start, end, surface, source_type in ordered:
        if any(a <= start and end <= b for a, b, _, _ in accepted):
            continue
        accepted.append((start, end, surface, source_type))
    return sorted(accepted, key=lambda item: (item[0], item[1], item[2]))


def classify_hard_features(
    surface: str,
    roster_alias_norms: set[str],
    ambiguous_norms: set[str],
) -> list[str]:
    """按与被测系统输出无关的表面与词法规则，判定难例层的抽样特征。"""
    s = surface.strip()
    norm = normalize_surface(s)
    features: list[str] = []
    if norm in ambiguous_norms:
        features.append("ambiguous_surface")
    if norm not in roster_alias_norms:
        features.append("out_of_roster")

    is_ascii = any("a" <= ch.lower() <= "z" for ch in s)
    if is_ascii:
        lower = norm.lower()
        tokens = [t.strip(".,;:!?\"'()[]—-") for t in lower.split() if t.strip(".,;:!?\"'()[]—-")]
        if any( lower.startswith(p) for p in EN_REL_PREFIXES ) or any(t in EN_REL_WORDS for t in tokens):
            features.append("kinship_or_relation")
        if any(t in EN_TITLE_WORDS or (t + ".") in EN_TITLE_WORDS for t in tokens):
            features.append("title_or_official")
        if len(tokens) == 1:
            features.append("surname_only_or_short")
    else:
        if 2 <= len(s) <= 4 and s.endswith("氏"):
            features.append("x_shi")
        if any(sub in s for sub in ZH_REL_SUBSTRS):
            features.append("kinship_or_relation")
        if any(sub in s for sub in ZH_TITLE_SUBSTRS):
            features.append("title_or_official")
        if 1 <= len(s) <= 2:
            features.append("surname_only_or_short")

    return sorted(set(features))


def allocate_proportional(counts: dict[str, int], target: int) -> dict[str, int]:
    """按各键出现频次比例分配名额 target（不设单别名 8 处上限，且不超过各自可用总数）。"""
    total = sum(counts.values())
    if total <= target:
        return dict(counts)
    raw = {k: (counts[k] * target) / total for k in counts}
    alloc = {k: min(counts[k], int(math.floor(raw[k]))) for k in counts}
    rem = target - sum(alloc.values())
    if rem > 0:
        order = sorted(
            counts,
            key=lambda k: (-(raw[k] - alloc[k]), -counts[k], k),
        )
        for k in order:
            if rem <= 0:
                break
            if alloc[k] < counts[k]:
                alloc[k] += 1
                rem -= 1
    while sum(alloc.values()) < target:
        progressed = False
        for k in sorted(counts, key=lambda k: (-counts[k], k)):
            if alloc[k] < counts[k] and sum(alloc.values()) < target:
                alloc[k] += 1
                progressed = True
        if not progressed:
            break
    return alloc


def extract_text_hard_spans(text: str, roster: dict, clean_hits: list[tuple[int, int, str, str]]) -> list[dict]:
    """当尚无发现数据库记录时（如留出短篇），按表面规则从原文提取难例候选位置。"""
    spans: dict[tuple[int, int], str] = {}
    for a, b, surf, _ in clean_hits:
        spans[(a, b)] = surf

    is_ascii_book = sum(1 for ch in text[:2000] if "a" <= ch.lower() <= "z") > 400
    if is_ascii_book:
        patterns = [
            r"\b(?:Mr\.|Mrs\.|Miss|Lady|Lord|Sir|Colonel|Inspector|Constable|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
            r"\b(?:the\s+Prime\s+Minister|the\s+Secretary\s+for\s+European\s+Affairs|the\s+Premier|his\s+Grace|her\s+Ladyship|Her\s+Ladyship)\b",
            r"\b(?:my|his|her|your|their)\s+(?:wife|husband|father|mother|sister|brother|friend|master|mistress|butler|maid|servant|colleague|client)\b",
            r"\bthe\s+(?:butler|maid|constable|inspector|minister|lady|woman|man|valet|secretary|messenger|police)\b",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                a, b = m.span()
                if not any(x <= a and b <= y for (x, y) in spans):
                    spans[(a, b)] = m.group(0)
    else:
        zh_terms = [
            "父亲", "母亲", "她的父亲", "她的叔父", "叔父", "叔远", "局长", "小东西",
            "油鸡", "官太太", "官太太们", "巡警", "房主", "朋友", "旧友", "女工",
            "车夫", "老东西", "叭儿狗", "会馆", "吉兆胡同", "教育部", "局里",
        ]
        for term in sorted(set(zh_terms), key=lambda x: (-len(x), x)):
            for a, b in exact_matches(text, term, 0, len(text)):
                if not any(x <= a and b <= y for (x, y) in spans):
                    spans[(a, b)] = term

    return [{"start": a, "end": b, "text": surf} for (a, b), surf in sorted(spans.items())]


def load_discovery_mentions(db_path: Path, run_id: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT data FROM records WHERE run_id=? AND kind='mention' ORDER BY start, id",
        (run_id,),
    ).fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]


def build_r3_candidates(
    text: str,
    roster: dict,
    chapters: list[dict],
    discovery_mentions: list[dict] | None = None,
    seed: int = 20261030,
    major_entity_samples: int = 12,
    hard_quota_long: int = 150,
    hard_quota_short: int = 80,
) -> dict:
    rng = random.Random(seed)
    L = len(text)
    is_long_book = L > 100000
    major_threshold = 20 if is_long_book else 5
    hard_quota = hard_quota_long if is_long_book else hard_quota_short

    # 若提供了共有发现提及，建立精确区间索引，优先对齐发现提及边界
    disc_by_exact: dict[tuple[int, int], dict] = {}
    if discovery_mentions:
        for dm in discovery_mentions:
            disc_by_exact[(dm["start"], dm["end"])] = dm

    alias_set: set[str] = set()
    roster_alias_norms: set[str] = set()
    entity_aliases: dict[str, set[str]] = {}
    for entity in roster.get("entities", []):
        gid = entity["gold_id"]
        entity_aliases[gid] = set()
        for alias in entity.get("aliases", []):
            if not alias or alias not in text:
                raise ValueError(f"名册别名未在原文出现：{alias}")
            alias_set.add(alias)
            roster_alias_norms.add(normalize_surface(alias))
            entity_aliases[gid].add(alias)

    ambiguous_set: set[str] = set()
    ambiguous_norms: set[str] = set()
    for item in roster.get("ambiguous_surfaces", []):
        surface = item["surface"]
        if not surface or surface not in text:
            raise ValueError(f"歧义称呼未在原文出现：{surface}")
        ambiguous_set.add(surface)
        ambiguous_norms.add(normalize_surface(surface))

    all_surfaces = sorted(alias_set | ambiguous_set, key=lambda s: (-len(s), s))
    raw_hits: list[tuple[int, int, str, str]] = []
    for surface in all_surfaces:
        stype = "ambiguous" if surface in ambiguous_set else "alias"
        for a, b in exact_matches(text, surface, 0, L):
            if _word_boundary(text, a, b, surface):
                raw_hits.append((a, b, surface, stype))

    clean_hits = deduplicate_longest_non_overlapping(raw_hits)
    hits_by_surface: dict[str, list[tuple[int, int, str, str]]] = collections.defaultdict(list)
    for hit in clean_hits:
        hits_by_surface[hit[2]].append(hit)

    # 1. 实体层（R3.2）：统计每个名册实体在原文中的出现次数，选出主要实体（长篇 >=20，短篇 >=5）
    entity_hit_pools: dict[str, list[tuple[int, int, str]]] = {}
    entity_occurrence_counts: dict[str, int] = {}
    major_entity_ids: list[str] = []

    for entity in roster.get("entities", []):
        gid = entity["gold_id"]
        # 统计该实体的非歧义专属别名出现位置（若实体仅有歧义别名则包含其全部别名位置）
        own_aliases = [a for a in entity.get("aliases", []) if a not in ambiguous_set]
        if not own_aliases:
            own_aliases = list(entity.get("aliases", []))
        pool: list[tuple[int, int, str]] = []
        for alias in own_aliases:
            for a, b, surf, _ in hits_by_surface.get(alias, []):
                pool.append((a, b, surf))
        pool = sorted(set(pool), key=lambda x: (x[0], x[1], x[2]))
        entity_hit_pools[gid] = pool
        entity_occurrence_counts[gid] = len(pool)
        if len(pool) >= major_threshold:
            major_entity_ids.append(gid)

    selected_spans: set[tuple[int, int]] = set()
    entity_stratum_items: list[dict] = []

    def third_index(pos: int) -> int:
        if pos < L / 3:
            return 0
        if pos < (2 * L) / 3:
            return 1
        return 2

    for entity in roster.get("entities", []):
        gid = entity["gold_id"]
        pool = entity_hit_pools[gid]
        if gid not in major_entity_ids:
            continue
        target = min(major_entity_samples, len(pool))
        if len(pool) <= target:
            chosen = list(pool)
        else:
            # 按前、中、后三段分层（每段目标 4 处，不足段余量按比例分给其余段），段内按别名出现比例分配
            by_third: dict[int, list[tuple[int, int, str]]] = {0: [], 1: [], 2: []}
            for item in pool:
                by_third[third_index(item[0])].append(item)
            third_counts = {str(t): len(by_third[t]) for t in (0, 1, 2) if by_third[t]}
            # 先给非空段平均分配（每段最多 4），若某段不足则由 allocate_proportional 补足
            base_alloc = {k: min(cnt, major_entity_samples // 3) for k, cnt in third_counts.items()}
            rem = target - sum(base_alloc.values())
            if rem > 0:
                rem_counts = {k: third_counts[k] - base_alloc[k] for k in third_counts if third_counts[k] > base_alloc[k]}
                extra = allocate_proportional(rem_counts, rem)
                for k, v in extra.items():
                    base_alloc[k] += v

            chosen = []
            for t_str, t_quota in sorted(base_alloc.items()):
                if t_quota <= 0:
                    continue
                t_pool = by_third[int(t_str)]
                by_alias: dict[str, list[tuple[int, int, str]]] = collections.defaultdict(list)
                for item in t_pool:
                    by_alias[item[2]].append(item)
                alias_counts = {al: len(items) for al, items in by_alias.items()}
                alias_alloc = allocate_proportional(alias_counts, t_quota)
                for al in sorted(alias_alloc):
                    k_take = alias_alloc[al]
                    al_list = list(by_alias[al])
                    # 优先选择与共有发现提及起止完全一致的位置（这不依赖任何身份分组，只看发现提及边界）
                    if disc_by_exact:
                        exact_pool = [it for it in al_list if (it[0], it[1]) in disc_by_exact]
                        other_pool = [it for it in al_list if (it[0], it[1]) not in disc_by_exact]
                        rng.shuffle(exact_pool)
                        rng.shuffle(other_pool)
                        al_shuffled = exact_pool + other_pool
                    else:
                        al_shuffled = list(al_list)
                        rng.shuffle(al_shuffled)
                    chosen.extend(al_shuffled[:k_take])

        for a, b, surf in sorted(chosen, key=lambda x: (x[0], x[1], x[2])):
            if (a, b) in selected_spans:
                continue
            selected_spans.add((a, b))
            entity_stratum_items.append({
                "start": a,
                "end": b,
                "surface": surf,
                "stratum": "entity",
                "target_major_entity": gid,
                "hard_features": [],
            })

    # 2. 难例层（R3.3）：从共有发现提及（或短篇原文规则候选）中按规则特征抽样
    raw_hard_pool = discovery_mentions if discovery_mentions is not None else extract_text_hard_spans(text, roster, clean_hits)
    hard_candidates_by_bucket: dict[int, list[dict]] = {1: [], 2: [], 3: []}

    for dm in raw_hard_pool:
        a, b = int(dm["start"]), int(dm["end"])
        surf = dm["text"]
        if (a, b) in selected_spans:
            continue
        if any(not (b <= x or a >= y) for (x, y) in selected_spans):
            continue
        feats = classify_hard_features(surf, roster_alias_norms, ambiguous_norms)
        if not feats:
            continue
        cand = {
            "start": a,
            "end": b,
            "surface": surf,
            "stratum": "hard",
            "target_major_entity": None,
            "hard_features": feats,
        }
        # 桶 1：关系/亲属描述、"X氏"、名册外称呼
        # 桶 2：歧义称呼、称谓/官职
        # 桶 3：仅姓氏、单字或双字短称呼
        if any(f in feats for f in ("kinship_or_relation", "x_shi", "out_of_roster")):
            hard_candidates_by_bucket[1].append(cand)
        elif any(f in feats for f in ("ambiguous_surface", "title_or_official")):
            hard_candidates_by_bucket[2].append(cand)
        else:
            hard_candidates_by_bucket[3].append(cand)

    bucket_targets = {
        1: int(round(hard_quota * 0.45)),
        2: int(round(hard_quota * 0.35)),
        3: hard_quota - int(round(hard_quota * 0.45)) - int(round(hard_quota * 0.35)),
    }
    surface_cap = 4 if is_long_book else 3
    surface_taken: collections.Counter[str] = collections.Counter()
    hard_stratum_items: list[dict] = []

    for b_id in (1, 2, 3):
        b_pool = list(hard_candidates_by_bucket[b_id])
        # 在桶内按前、中、后三段交错洗牌，保证全书分布均匀
        thirds = {0: [], 1: [], 2: []}
        for c in b_pool:
            thirds[third_index(c["start"])].append(c)
        for t in (0, 1, 2):
            rng.shuffle(thirds[t])
        interleaved: list[dict] = []
        max_len = max((len(v) for v in thirds.values()), default=0)
        for i in range(max_len):
            for t in (0, 1, 2):
                if i < len(thirds[t]):
                    interleaved.append(thirds[t][i])

        b_need = bucket_targets[b_id]
        b_got = 0
        for c in interleaved:
            if b_got >= b_need or len(hard_stratum_items) >= hard_quota:
                break
            span = (c["start"], c["end"])
            if span in selected_spans or any(not (c["end"] <= x or c["start"] >= y) for (x, y) in selected_spans):
                continue
            norm = normalize_surface(c["surface"])
            if surface_taken[norm] >= surface_cap:
                continue
            selected_spans.add(span)
            surface_taken[norm] += 1
            hard_stratum_items.append(c)
            b_got += 1

    # 若某些桶不满导致难例总数未达 hard_quota，从剩余难例候选中补齐（首先遵守 surface_cap，仍不足再放宽）
    if len(hard_stratum_items) < hard_quota:
        remaining = [c for b_id in (1, 2, 3) for c in hard_candidates_by_bucket[b_id] if (c["start"], c["end"]) not in selected_spans]
        rng.shuffle(remaining)
        for relax in (False, True):
            for c in remaining:
                if len(hard_stratum_items) >= hard_quota:
                    break
                span = (c["start"], c["end"])
                if span in selected_spans or any(not (c["end"] <= x or c["start"] >= y) for (x, y) in selected_spans):
                    continue
                norm = normalize_surface(c["surface"])
                if not relax and surface_taken[norm] >= surface_cap:
                    continue
                selected_spans.add(span)
                surface_taken[norm] += 1
                hard_stratum_items.append(c)

    all_items = sorted(entity_stratum_items + hard_stratum_items, key=lambda x: (x["start"], x["end"], x["surface"]))
    candidates = []
    for idx, item in enumerate(all_items, start=1):
        start, end = item["start"], item["end"]
        ch_idx = find_chapter_index(chapters, start)
        candidates.append({
            "sample_id": f"s{idx:03d}",
            "start": start,
            "end": end,
            "surface": item["surface"],
            "stratum": item["stratum"],
            "hard_features": item["hard_features"],
            "chapter_index": ch_idx,
            "before": text[max(0, start - 300):start],
            "after": text[end:min(L, end + 300)],
            "gold_id": None,
            "note": "",
        })

    return {
        "book": roster["book"],
        "book_id": roster.get("book_id", ""),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "seed": seed,
        "sampling_version": "r3-stratified-v1",
        "major_entity_min_occurrences": major_threshold,
        "major_entity_ids": major_entity_ids,
        "entity_occurrence_counts": entity_occurrence_counts,
        "entity_stratum_count": len(entity_stratum_items),
        "hard_stratum_count": len(hard_stratum_items),
        "total_candidates": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "test/text-library/reference/source-manifest.json")
    parser.add_argument("--discovery-db", type=Path, default=None)
    parser.add_argument("--discovery-run", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    raw = args.text.read_bytes()
    text = raw.removeprefix(b"\xef\xbb\xbf").decode("utf-8")
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    chapters = []
    for book in manifest.get("books", []):
        if Path(book["filename"]).name == args.text.name or book["sha256"] == roster["source_sha256"]:
            chapters = book.get("chapters", [])
            break

    disc_mentions = None
    if args.discovery_db and args.discovery_run:
        disc_mentions = load_discovery_mentions(args.discovery_db, args.discovery_run)

    result = build_r3_candidates(
        text,
        roster,
        chapters,
        discovery_mentions=disc_mentions,
        seed=args.seed,
    )
    write_json(args.output, result)
    print(json.dumps({
        "book": result["book"],
        "major_entities": len(result["major_entity_ids"]),
        "entity_stratum": result["entity_stratum_count"],
        "hard_stratum": result["hard_stratum_count"],
        "total_candidates": result["total_candidates"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
