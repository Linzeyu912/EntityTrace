"""实体恒定底色继承与阶段内时间序场景造型快照模块（纯函数、零 IO、零网络请求）。

在不修改任何既有 `facts` / `state_events` / `stages[].facts` / `stages[].gaps` 字段的前提下，
按原文时间顺序为实体增量构建：
1. 跨时期恒定底色（`base_traits`）及后续阶段继承底色（`stages[].inherited_traits`）与真实空缺（`stages[].effective_gaps`）；
2. 每个宏观阶段内严格时间单调、事实守恒的场景造型快照列表（`stages[].scene_appearances`）。
"""
from __future__ import annotations

import collections
import copy
import re
from typing import Any

from .life_stages import _DISGUISE_RE, _build_body_seq

_EXTENDED_DISGUISE_RE = re.compile(
    r"(?:幻化出|幻化成|幻化为|变幻成|化作一名|化作一个|易容|伪装|假扮|乔装|掩饰|"
    r"并非.{0,4}真正容貌|不是.{0,8}真正容貌|假相|\bdisguis(?:e|ed)\b|\bin\s+disguise\b)",
    re.IGNORECASE,
)

_NON_TRAIT_PERSON_STATE_RE = re.compile(
    r"(?:发青|苍老|病容|黑气|幻化|三头八臂|三首八臂|面如蓝靛|发似朱砂|"
    r"金属光泽|银白色|血芒|鬼爪|面具|绝气|老态|伤|血|青紫|肿|惨白|苍白|"
    r"失声|嘶哑了许多|沙哑了许多|兽性吼声|满面通红|瘦了|憔悴|死灰|灰败|"
    r"\b(?:blush(?:ed|ing)?|pale|pallid|flush(?:ed|ing)?|trembling)\b)",
    re.IGNORECASE,
)

_NON_TRAIT_OBJECT_STATE_RE = re.compile(
    r"(?:断|碎|裂|毁|残|折断|崩塌|焚毁|化作|受损|broken|shattered|cracked|damaged)",
    re.IGNORECASE,
)

# 人物装束核心品类（秋冬皮毛厚袄独立于常服，避免与常服混堆）
_COSTUME_CATEGORY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("prison", re.compile(r"护身枷|盘头枷|戴枷|铁枷|木枷|囚服|囚衣|刑具|\bchains?\b|\bshackles?\b", re.I)),
    ("mourning", re.compile(r"孝服|丧服|缟素|孝绢|麻衣孝|穿白挂素|\bmourning\b", re.I)),
    ("bare", re.compile(r"赤裸|赤膊|赤条条|不著一丝|光着上身|脱去上衣|只穿.{0,4}内裤|\bnaked\b|\bshirtless\b|\bbare-chested\b", re.I)),
    ("tattered", re.compile(r"破烂|褴褛|破布衫|衣甲飘零|冠歪袍绽|头巾破碎|\btattered\b|\bragged\b", re.I)),
    ("armor", re.compile(r"铠甲|甲胄|金甲|银甲|铁甲|白甲|红甲|锁甲|锁子甲|连环甲|乌油甲|全装甲|戎装|盔甲|披挂|战袍|\barmor\b|\barmour\b", re.I)),
    ("taoist_monk", re.compile(r"道服|道袍|鹤氅|八卦仙衣|直裰|直缀|僧衣|袈裟|羽服|水合服|水合袍|九宫衣服|\brobes?\s+of\s+a\s+monk\b", re.I)),
    ("official", re.compile(r"公服|朝服|王服|衮服|冕旒|官服|\bofficial\s+robe\b|\bcourt\s+dress\b", re.I)),
    ("disguise", re.compile(r"货郎|扮作|乔装|\bdisguise\b", re.I)),
    ("winter_fur", re.compile(r"皮袄|貂鼠|狐裘|貂裘|锦裘|皮裘|羊皮袄|棉袄|\bfur\s+coat\b", re.I)),
    ("casual", re.compile(
        r"布衫|青衫|黄衫|白衫|罗衫|长衫|短打|汗衫|灰袍|蓝袍|黑袍|绿袍|红袍|白袍|黄袍|锦袍|花袍|绣袍|长袍|团领|袍袄|"
        r"紫衣|白衣|黄衣|蓝衣|黑衣|红衣|绿衣|青衣|素净衣服|紧身衣衫|衣衫|衣裙|白裙|黑裙|衲袄|缎子袄|段子袄|"
        r"衣服|衣裳|衣着|装束|打扮|"
        r"短褐|弟子服饰|门内服饰|七玄门衣衫|黄枫谷服饰|门派服饰|夹克|外套|外衣|衬衫|上衣|中式上衣|基督徒的衣服|"
        r"\b(?:coat|jacket|shirt|dress|gown|tunic|robe|cloak|garment|suit|clothes|outfit|attire)\b",
        re.I,
    )),
)

_STRONG_STATE_COSTUME_CATS = frozenset({
    "prison", "mourning", "bare", "tattered", "armor", "disguise",
})

_PORTRAIT_LAYER_CATS = frozenset({
    "armor", "casual", "official", "winter_fur",
})

# 服装外层披风（可与内穿长袍共存，不与主长袍互斥）
_OUTER_CLOAK_RE = re.compile(r"斗篷|披风|蓑衣|斗笠|面纱|轻纱")

# 颜色词根（仅当该事实含主服装品类时参与色系互斥判断）
_COSTUME_COLOR_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("black", re.compile(r"漆黑|黑衣|黑袍|黑裙|黑色|乌油|皂布|皂直|皂罗|皂衫|玄色|墨色|\bblack\b", re.I)),
    ("white", re.compile(r"白袍|白甲|白衣|白衫|白裙|白缎|白段|白色|白娟|白绢|缟素|素净|雪白|\bwhite\b", re.I)),
    ("red", re.compile(r"大红|红袍|红甲|红衣|红锦|红䌷|红裙|绯红|绛绡|赤红|朱红|烈火|穿红|\bred\b|\bcrimson\b|\bscarlet\b", re.I)),
    ("yellow", re.compile(r"黄衫|黄袍|淡黄|杏黄|土黄|土色|赭黄|黄色|金甲|浑金甲|金锁|黄金甲|金装|\byellow\b|\bgold(?:en)?\b", re.I)),
    ("cyan", re.compile(r"青衫|青袍|青衣|青纱|青团领|深青|\bcyan\b", re.I)),
    ("green", re.compile(r"绿袍|绿衣|绿罗|绿锦|鹦哥绿|淡绿|绿色|翡翠|\bgreen\b", re.I)),
    ("blue", re.compile(r"蓝衣|蓝袍|蓝色|\bblue\b", re.I)),
    ("grey", re.compile(r"灰袍|灰色|茶褐|褐衲|浅褐|\bgrey\b|\bgray\b|\bbrown\b", re.I)),
    ("purple", re.compile(r"紫衣|紫袍|紫绣|紫色|\bpurple\b", re.I)),
)

# 地点昼夜／季节／气象／存毁场景状态分类
_PLACE_ATMOS_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("night", re.compile(r"夜|月光|月色|星光|灯火|烛光|黑暗|漆黑|黄昏|暮色|\bnight\b|\bmoonlight\b|\bdarkness\b", re.I)),
    ("day", re.compile(r"白昼|白天|烈日|阳光|晴朗|晨光|朝阳|旭日|\bdaylight\b|\bsunlight\b|\bmorning\b", re.I)),
    ("winter_snow", re.compile(r"风雪|大雪|积雪|冰天雪地|寒冬|严寒|雪花|\bsnow\b|\bwinter\b|\bfrost\b", re.I)),
    ("storm_rain", re.compile(r"暴雨|雷雨|狂风|乌云|电闪雷鸣|阴雨|\bstorm\b|\bthunder\b|\brain\b", re.I)),
    ("ruined", re.compile(r"残垣|废墟|荒废|焦土|灰烬|血腥|战火|火光冲天|倒塌|破败|\bruin(?:ed|s)?\b|\bashes\b|\bflames?\b", re.I)),
)

_SCENE_TRIGGER_EVENT_TYPES = frozenset({
    "costume_change", "disguise", "transform", "injury", "damage", "owner_change", "upgrade",
})


def _normalize_text_key(s: str) -> str:
    return re.sub(
        r"[\s\u3000·•.,，、。！？；：\"'“”‘’()（）\[\]【】]+",
        "",
        str(s or "").strip(),
    ).lower()


def parse_costume_signature(value: str, quote: str = "") -> dict[str, Any]:
    """解析单条 person.costume 事实的服装品类与主色系签名（用于互斥装束检测与同造型合并）。"""
    txt = f"{value or ''}"
    cats: set[str] = set()
    for cname, pat in _COSTUME_CATEGORY_RULES:
        if pat.search(txt):
            cats.add(cname)
    if not cats and quote:
        for cname, pat in _COSTUME_CATEGORY_RULES:
            if pat.search(quote):
                cats.add(cname)
    is_cloak_only = bool(_OUTER_CLOAK_RE.search(txt)) and not cats
    is_accessory_only = (not cats) and (not is_cloak_only)
    colors: set[str] = set()
    if cats:
        for col, pat in _COSTUME_COLOR_RULES:
            if pat.search(txt):
                colors.add(col)
    return {
        "categories": cats,
        "colors": colors,
        "is_cloak_only": is_cloak_only,
        "is_accessory_only": is_accessory_only,
    }


def costumes_conflict(sig_a: dict[str, Any], sig_b: dict[str, Any]) -> bool:
    """判定两条装束事实是否属于互斥装束（不同服装体系、单侧强状态品类或不同主色系）。"""
    cats_a, cats_b = sig_a["categories"], sig_b["categories"]
    if not cats_a or not cats_b:
        return False
    if not (cats_a & cats_b):
        return True
    if (cats_a ^ cats_b) & _STRONG_STATE_COSTUME_CATS:
        return True
    cols_a, cols_b = sig_a["colors"], sig_b["colors"]
    if cols_a and cols_b and not (cols_a & cols_b):
        return True
    return False


def count_costume_conflict_pairs(
    facts: list[dict],
    body_seq: dict[str, int] | None = None,
    *,
    portrait_window: int = 260,
) -> int:
    """统计一组事实内互相冲突的装束对数（排除同章紧邻全身肖像叠穿如袍+甲／裘袄）。"""
    c_facts = [f for f in facts if f.get("facet") == "costume"]
    if len(c_facts) < 2:
        return 0
    sigs = [
        parse_costume_signature(str(f.get("value") or ""), str(f.get("quote") or ""))
        for f in c_facts
    ]
    conflicts = 0
    for i in range(len(c_facts)):
        for j in range(i + 1, len(c_facts)):
            if not costumes_conflict(sigs[i], sigs[j]):
                continue
            fi, fj = c_facts[i], c_facts[j]
            same_ch = fi.get("chapter_id") is not None and fi.get("chapter_id") == fj.get("chapter_id")
            if same_ch and abs(int(fj["start"]) - int(fi["end"])) <= portrait_window:
                union_cats = sigs[i]["categories"] | sigs[j]["categories"]
                if union_cats <= _PORTRAIT_LAYER_CATS:
                    continue
            conflicts += 1
    return conflicts


def _parse_place_atmos_tags(value: str, quote: str = "") -> set[str]:
    txt = f"{value or ''} {quote or ''}"
    tags: set[str] = set()
    for tag, pat in _PLACE_ATMOS_RULES:
        if pat.search(txt):
            tags.add(tag)
    return tags


def _place_atmos_conflict(tags_a: set[str], tags_b: set[str]) -> bool:
    if not tags_a or not tags_b:
        return False
    pairs = (
        ("day", "night"),
        ("winter_snow", "day"),
        ("ruined", "day"),
    )
    for x, y in pairs:
        if (x in tags_a and y in tags_b) or (y in tags_a and x in tags_b):
            return True
    if "ruined" in (tags_a ^ tags_b) and (
        tags_a & {"day", "night", "winter_snow", "storm_rain"}
        or tags_b & {"day", "night", "winter_snow", "storm_rain"}
    ):
        return True
    return False


def _holder_records_from_facts(facts: list[dict]) -> list[tuple[str | None, str]]:
    recs: list[tuple[str | None, str]] = []
    for f in facts:
        if f.get("facet") != "holder":
            continue
        hid_raw = f.get("holder_entity_id")
        hid = str(hid_raw).strip() if isinstance(hid_raw, str) and hid_raw.strip() else None
        nv = _normalize_text_key(str(f.get("value") or ""))
        if hid or nv:
            recs.append((hid, nv))
    return recs


def _holders_conflict(
    prev_recs: list[tuple[str | None, str]],
    curr_recs: list[tuple[str | None, str]],
) -> bool:
    """判定两组 holder 事实是否明确指向不同持有者（兼容一侧缺 holder_entity_id 的同名情况）。"""
    if not prev_recs or not curr_recs:
        return False
    prev_ids = {hid for hid, _ in prev_recs if hid}
    curr_ids = {hid for hid, _ in curr_recs if hid}
    if prev_ids and curr_ids:
        return not bool(prev_ids & curr_ids)
    prev_names = {nv for _, nv in prev_recs if nv}
    curr_names = {nv for _, nv in curr_recs if nv}
    if not prev_names or not curr_names:
        return False
    for pn in prev_names:
        for cn in curr_names:
            if pn == cn or pn in cn or cn in pn:
                return False
    return True


def _select_base_traits(
    entity: dict,
    stages: list[dict],
    chapters: list[dict] | None = None,
) -> list[dict]:
    """提取跨时期恒定底色事实（优先 persistence == 'trait'，无标签时按粗类启发式提取）。"""
    kind = str(entity.get("kind") or "person")
    facts = list(entity.get("facts") or [])
    events = list(entity.get("state_events") or [])
    if not facts:
        return []

    has_persistence = any(f.get("persistence") in ("trait", "stage", "scene") for f in facts)
    if has_persistence:
        traits = [f for f in facts if f.get("persistence") == "trait"]
        traits.sort(key=lambda f: (int(f["start"]), int(f["end"]), str(f.get("id", ""))))
        return traits

    chapters_list = list(chapters or [])
    body_seq = _build_body_seq(chapters_list)
    ch_by_id = {str(ch["id"]): ch for ch in chapters_list}

    fact_stage_map: dict[str, int] = {}
    for st in stages:
        sidx = int(st["stage_index"])
        for f in st.get("facts") or []:
            fact_stage_map[str(f["id"])] = sidx

    # 计算器物／地点的首个损毁或变形切分点；若更早章节已有固有维度事实，前移至剧变章起点
    first_alter_ev: dict | None = None
    first_alter_start: int | None = None
    for ev in events:
        if ev.get("stage_split") is True and ev.get("event_type") in ("damage", "transform"):
            s = int(ev["start"])
            if first_alter_start is None or s < first_alter_start:
                first_alter_start = s
                first_alter_ev = ev

    if first_alter_ev is not None and kind in ("object", "place"):
        inherent_facets = (
            ("shape", "size", "color_material") if kind == "object" else ("terrain", "scale")
        )
        ev_cid = first_alter_ev.get("chapter_id")
        ev_ch_ord = body_seq.get(str(ev_cid)) if ev_cid is not None else None
        ch_obj = ch_by_id.get(str(ev_cid)) if ev_cid is not None else None
        if ev_ch_ord is not None and ch_obj is not None:
            has_earlier_inherent = False
            for f in facts:
                if f.get("facet") not in inherent_facets:
                    continue
                if kind == "object" and _NON_TRAIT_OBJECT_STATE_RE.search(str(f.get("value") or "")):
                    continue
                fcid = f.get("chapter_id")
                f_ord = body_seq.get(str(fcid)) if fcid is not None else None
                if f_ord is not None and f_ord < ev_ch_ord:
                    has_earlier_inherent = True
                    break
            if has_earlier_inherent:
                first_alter_start = min(int(first_alter_ev["start"]), int(ch_obj["start"]))

    traits: list[dict] = []
    if kind == "person":
        build_nums_by_stage: dict[int, set[str]] = collections.defaultdict(set)
        for f in facts:
            if f.get("facet") == "build":
                m_h = re.search(r"[一二两三四五六七八九十百\d]+(?:丈|尺)", str(f.get("value") or ""))
                if m_h:
                    build_nums_by_stage[fact_stage_map.get(str(f["id"]), 1)].add(m_h.group(0))
        build_changed_across_stages = (
            len({v for vs in build_nums_by_stage.values() for v in vs}) >= 2
            and len(build_nums_by_stage) >= 2
        )

        for f in facts:
            fc = f.get("facet")
            val = str(f.get("value") or "")
            quo = str(f.get("quote") or "")
            if fc == "gender":
                traits.append(f)
            elif fc == "voice":
                if not _NON_TRAIT_PERSON_STATE_RE.search(val) and not _NON_TRAIT_PERSON_STATE_RE.search(quo):
                    traits.append(f)
            elif fc in ("build", "face", "feature"):
                if fc == "build" and build_changed_across_stages:
                    continue
                if _NON_TRAIT_PERSON_STATE_RE.search(val) or _NON_TRAIT_PERSON_STATE_RE.search(quo):
                    continue
                if _DISGUISE_RE.search(val) or _DISGUISE_RE.search(quo) or _EXTENDED_DISGUISE_RE.search(quo):
                    continue
                traits.append(f)
    elif kind == "object":
        for f in facts:
            if f.get("facet") in ("shape", "size", "color_material"):
                if first_alter_start is not None and int(f["start"]) >= first_alter_start:
                    continue
                if _NON_TRAIT_OBJECT_STATE_RE.search(str(f.get("value") or "")):
                    continue
                traits.append(f)
    elif kind == "place":
        for f in facts:
            if f.get("facet") in ("terrain", "scale"):
                if first_alter_start is not None and int(f["start"]) >= first_alter_start:
                    continue
                traits.append(f)

    traits.sort(key=lambda f: (int(f["start"]), int(f["end"]), str(f.get("id", ""))))
    return traits


def build_entity_timeline_profiles(
    entity: dict,
    chapters: list[dict],
    *,
    text: str | None = None,
) -> dict:
    """在不修改任何原有字段的前提下，为实体附加跨时期恒定底色继承与阶段内时间序场景造型快照。"""
    from .describe import FACETS

    out = copy.deepcopy(entity)
    kind = str(out.get("kind") or "person")
    stages = list(out.get("stages") or [])
    all_events = sorted(
        list(out.get("state_events") or []),
        key=lambda ev: (int(ev["start"]), int(ev["end"]), str(ev.get("id", ""))),
    )
    chapters_list = list(chapters or [])
    body_seq = _build_body_seq(chapters_list)

    base_traits = _select_base_traits(out, stages, chapters_list)
    base_trait_ids = {str(f["id"]) for f in base_traits}
    out["base_traits"] = base_traits

    prior_base_traits: list[dict] = []
    total_appearances = 0

    for m, stage in enumerate(stages):
        st_idx = int(stage["stage_index"])
        st_start = int(stage["start"])
        st_end = int(stage["end"])
        st_facts = sorted(
            list(stage.get("facts") or []),
            key=lambda f: (int(f["start"]), int(f["end"]), str(f.get("id", ""))),
        )
        st_fact_facets = {str(f["facet"]) for f in st_facts}

        # 1. 阶段继承底色与扣除继承后的真实空缺 effective_gaps
        if m == 0:
            inherited_traits: list[dict] = []
        else:
            inherited_traits = [
                f for f in prior_base_traits
                if str(f["facet"]) not in st_fact_facets
            ]
        eff_covered = st_fact_facets | {str(f["facet"]) for f in inherited_traits}
        effective_gaps = [fc for fc in FACETS.get(kind, ()) if fc not in eff_covered]

        stage["inherited_traits"] = inherited_traits
        stage["effective_gaps"] = effective_gaps

        # 2. 筛选落入本阶段的保留状态变化事件
        is_last_stage = (m == len(stages) - 1)
        st_events = [
            ev for ev in all_events
            if (int(ev["start"]) >= st_start) and (is_last_stage or int(ev["start"]) < st_end)
        ]

        st_own_traits = [f for f in st_facts if str(f["id"]) in base_trait_ids]
        st_scene_facts = [f for f in st_facts if str(f["id"]) not in base_trait_ids]

        # 3. 空阶段
        if not st_facts and not st_events:
            stage["scene_appearances"] = []
            continue

        # 4. 合并本阶段场景事实与事件为重叠原子时间簇（确保 prev.end <= curr.start）
        items: list[dict] = []
        for ev in st_events:
            items.append({
                "kind": "event",
                "start": int(ev["start"]),
                "end": int(ev["end"]),
                "ch_ord": body_seq.get(str(ev.get("chapter_id"))),
                "ch_id": ev.get("chapter_id"),
                "obj": ev,
            })
        for f in st_scene_facts:
            items.append({
                "kind": "fact",
                "start": int(f["start"]),
                "end": int(f["end"]),
                "ch_ord": body_seq.get(str(f.get("chapter_id"))),
                "ch_id": f.get("chapter_id"),
                "obj": f,
            })

        if not items:
            ords = [
                body_seq[str(f.get("chapter_id"))]
                for f in st_own_traits
                if str(f.get("chapter_id")) in body_seq
            ]
            s_lo = min(int(f["start"]) for f in st_own_traits)
            s_hi = max(int(f["end"]) for f in st_own_traits)
            stage["scene_appearances"] = [{
                "appearance_index": 1,
                "stage_index": st_idx,
                "appearance_code": f"S{st_idx}-L{1}",
                "from_chapter": min(ords) if ords else stage.get("from_chapter"),
                "to_chapter": max(ords) if ords else stage.get("to_chapter"),
                "start": s_lo,
                "end": s_hi,
                "trigger_kind": "stage_initial",
                "trigger_summary": "阶段初始造型（基础底色特征）",
                "scene_facts": [],
                "active_traits": sorted(
                    [*inherited_traits, *st_own_traits],
                    key=lambda f: (int(f["start"]), int(f["end"]), str(f.get("id", ""))),
                ),
                "scene_events": [],
            }]
            total_appearances += 1
            prior_base_traits.extend(st_own_traits)
            continue

        items.sort(
            key=lambda it: (
                it["start"],
                0 if it["kind"] == "event" else 1,
                it["end"],
                str(it["obj"].get("id", "")),
            )
        )

        clusters: list[dict] = []
        for it in items:
            if clusters and it["start"] < clusters[-1]["end"]:
                cl = clusters[-1]
                cl["end"] = max(cl["end"], it["end"])
                if it["kind"] == "event":
                    cl["events"].append(it["obj"])
                else:
                    cl["facts"].append(it["obj"])
                if it["ch_ord"] is not None:
                    cl["ch_ords"].append(it["ch_ord"])
                if it["ch_id"] is not None:
                    cl["ch_ids"].add(it["ch_id"])
            else:
                clusters.append({
                    "start": it["start"],
                    "end": it["end"],
                    "events": [it["obj"]] if it["kind"] == "event" else [],
                    "facts": [it["obj"]] if it["kind"] == "fact" else [],
                    "ch_ords": [it["ch_ord"]] if it["ch_ord"] is not None else [],
                    "ch_ids": {it["ch_id"]} if it["ch_id"] is not None else set(),
                })

        # 5. 逐原子时间簇决定延续合并或开启新场景造型快照
        snaps: list[dict] = []

        def _new_snap(cl: dict, trig_kind: str, trig_summary: str) -> dict:
            return {
                "start": cl["start"],
                "end": cl["end"],
                "ch_ords": list(cl["ch_ords"]),
                "ch_ids": set(cl["ch_ids"]),
                "trigger_kind": trig_kind,
                "trigger_summary": trig_summary,
                "scene_facts": list(cl["facts"]),
                "scene_events": list(cl["events"]),
                "costume_sigs": [
                    (f, parse_costume_signature(str(f.get("value") or ""), str(f.get("quote") or "")))
                    for f in cl["facts"]
                    if f.get("facet") == "costume"
                ],
                "holders": _holder_records_from_facts(cl["facts"]),
                "atmos_tags": set().union(*(
                    _parse_place_atmos_tags(str(f.get("value") or ""), str(f.get("quote") or ""))
                    for f in cl["facts"]
                    if f.get("facet") == "atmosphere"
                )) if cl["facts"] else set(),
                "last_event_types": {ev.get("event_type") for ev in cl["events"]},
            }

        for cl in clusters:
            if not snaps:
                init_ev = cl["events"][0] if cl["events"] else None
                init_sum = (
                    f"阶段初始造型（{init_ev['event_type']}:{init_ev['value']}）"
                    if init_ev else "阶段初始造型"
                )
                snaps.append(_new_snap(cl, "stage_initial", init_sum))
                continue

            cur = snaps[-1]
            cur_last_ch = max(cur["ch_ords"]) if cur["ch_ords"] else None
            cl_first_ch = min(cl["ch_ords"]) if cl["ch_ords"] else None
            same_ch = (
                cur_last_ch is not None
                and cl_first_ch is not None
                and cl_first_ch == cur_last_ch
            )

            split_reason: tuple[str, str] | None = None

            # 条件 1：当场保留状态变化事件触发
            trig_evs = [
                ev for ev in cl["events"] if ev.get("event_type") in _SCENE_TRIGGER_EVENT_TYPES
            ]
            if trig_evs:
                ev0 = trig_evs[0]
                etype = str(ev0["event_type"])
                is_same_ch_repeat_injury = (
                    same_ch
                    and etype == "injury"
                    and "injury" in cur["last_event_types"]
                    and not cl["facts"]
                )
                if not is_same_ch_repeat_injury:
                    split_reason = (f"scene_event:{etype}", f"{etype}：{ev0['value']}")

            # 条件 2：跨章／同章互斥装束、持有者、场景气象切换
            if split_reason is None and kind == "person":
                cl_costumes = [f for f in cl["facts"] if f.get("facet") == "costume"]
                for cf in cl_costumes:
                    csig = parse_costume_signature(
                        str(cf.get("value") or ""), str(cf.get("quote") or ""),
                    )
                    conflict_hit = False
                    for prev_f, prev_sig in cur["costume_sigs"]:
                        if not costumes_conflict(prev_sig, csig):
                            continue
                        if same_ch and abs(int(cf["start"]) - int(prev_f["end"])) <= 260:
                            if (prev_sig["categories"] | csig["categories"]) <= _PORTRAIT_LAYER_CATS:
                                continue
                        conflict_hit = True
                        break
                    if conflict_hit:
                        split_reason = ("costume_switch", f"装束更替：{cf['value']}")
                        break

            if split_reason is None and kind == "object" and not same_ch:
                cl_holders = _holder_records_from_facts(cl["facts"])
                if _holders_conflict(cur["holders"], cl_holders):
                    hf = next(f for f in cl["facts"] if f.get("facet") == "holder")
                    split_reason = ("holder_switch", f"持有者切换：{hf['value']}")

            if split_reason is None and kind == "place":
                cl_atmos: set[str] = set()
                for f in cl["facts"]:
                    if f.get("facet") == "atmosphere":
                        cl_atmos |= _parse_place_atmos_tags(
                            str(f.get("value") or ""), str(f.get("quote") or ""),
                        )
                if _place_atmos_conflict(cur["atmos_tags"], cl_atmos):
                    af = next(f for f in cl["facts"] if f.get("facet") == "atmosphere")
                    split_reason = ("atmosphere_switch", f"场景气象/状态切换：{af['value']}")

            # 条件 3：长间隔独立重登场（排除仅含同阶段 age 事实的空壳切分）
            if (
                split_reason is None
                and cur_last_ch is not None
                and cl_first_ch is not None
                and (cl_first_ch - cur_last_ch) >= 10
                and (bool(cl["events"]) or any(f.get("facet") != "age" for f in cl["facts"]))
            ):
                gap_len = cl_first_ch - cur_last_ch
                split_reason = ("reappearance_gap", f"隔 {gap_len} 章独立重登场（第 {cl_first_ch} 章）")

            if split_reason is not None:
                snaps.append(_new_snap(cl, split_reason[0], split_reason[1]))
            else:
                cur["end"] = max(cur["end"], cl["end"])
                cur["ch_ords"].extend(cl["ch_ords"])
                cur["ch_ids"].update(cl["ch_ids"])
                cur["scene_facts"].extend(cl["facts"])
                cur["scene_events"].extend(cl["events"])
                for f in cl["facts"]:
                    if f.get("facet") == "costume":
                        cur["costume_sigs"].append(
                            (f, parse_costume_signature(str(f.get("value") or ""), str(f.get("quote") or "")))
                        )
                    elif f.get("facet") == "holder":
                        cur["holders"].extend(_holder_records_from_facts([f]))
                    elif f.get("facet") == "atmosphere":
                        cur["atmos_tags"] |= _parse_place_atmos_tags(
                            str(f.get("value") or ""), str(f.get("quote") or ""),
                        )
                cur["last_event_types"].update(ev.get("event_type") for ev in cl["events"])

        # 6. 装配输出快照并挂载 active_traits
        final_apps: list[dict] = []
        for r_idx, sp in enumerate(snaps, start=1):
            is_last_snap = (r_idx == len(snaps))
            active_own = [
                f for f in st_own_traits
                if is_last_snap or int(f["start"]) <= int(sp["end"])
            ]
            active_traits = sorted(
                [*inherited_traits, *active_own],
                key=lambda f: (int(f["start"]), int(f["end"]), str(f.get("id", ""))),
            )
            ch_ords = sp["ch_ords"]
            final_apps.append({
                "appearance_index": r_idx,
                "stage_index": st_idx,
                "appearance_code": f"S{st_idx}-L{r_idx}",
                "from_chapter": min(ch_ords) if ch_ords else stage.get("from_chapter"),
                "to_chapter": max(ch_ords) if ch_ords else stage.get("to_chapter"),
                "start": int(sp["start"]),
                "end": int(sp["end"]),
                "trigger_kind": sp["trigger_kind"],
                "trigger_summary": sp["trigger_summary"],
                "scene_facts": sorted(
                    sp["scene_facts"],
                    key=lambda f: (int(f["start"]), int(f["end"]), str(f.get("id", ""))),
                ),
                "active_traits": active_traits,
                "scene_events": sorted(
                    sp["scene_events"],
                    key=lambda ev: (int(ev["start"]), int(ev["end"]), str(ev.get("id", ""))),
                ),
            })

        stage["scene_appearances"] = final_apps
        total_appearances += len(final_apps)
        prior_base_traits.extend(st_own_traits)

    out["total_scene_appearances"] = total_appearances
    return out
