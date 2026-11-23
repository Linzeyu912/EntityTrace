"""跨时期多马甲／别号疑似同人待审提示模块（纯函数、零 IO、零网络请求）。

融合三路确定性信号生成低打扰、可逐字回溯原文的只读待审提示（`persona_alias_hints`），
绝不自动改写名册分组或突破不可合并约束：
1. `c2_blocked_same`：C2 补合并连通块因局部冲突整组拦截的人物对；
2. `explicit_alias_text`：原文同句紧邻本名／化名／改名／表字／道号显式揭示（内置亲属、师徒、敌对、互动与让步连词否决守卫）；
3. `complementary_chapters_shared_anchor`：活跃章节高度互补（重叠占比 <= 0.15 且早晚接替）且共享非易主专名持有物或全书罕见标志性特征的人物对。
"""
from __future__ import annotations

import collections
import math
import re

from .life_stages import _build_body_seq
from .util import digest

_GENERIC_PERSON_SURFACES = frozenset({
    "老者", "老人", "老翁", "老妇", "老汉", "老头", "老道", "老僧",
    "少年", "少女", "青年", "男子", "女子", "汉子", "大汉", "壮汉",
    "妇人", "夫人", "小姐", "公子", "书生", "道士", "道人", "道长", "和尚", "僧人",
    "先生", "大夫", "郎中", "员外", "秀才", "官人", "相公", "大师", "师太",
    "大人", "老爷", "太太", "奶奶", "姑娘", "婆婆", "婆子", "小二", "店家", "掌柜",
    "侍女", "丫鬟", "仆人", "护卫", "士兵", "将领", "将军", "元帅", "太师", "丞相",
    "众人", "那人", "此人", "那厮", "这厮", "来人", "敌人", "对手", "修士",
    "掌门", "长老", "宗主", "帮主", "寨主", "教主", "头领", "头目", "首领", "大王",
    "弟子", "徒弟", "师父", "师兄", "师弟", "师姐", "师妹", "前辈", "晚辈", "仙子",
    "皇帝", "天子", "国王", "大臣", "王爷", "太子", "公主", "娘娘", "皇后", "太后",
    "县官", "知府", "太守", "知县",
    "man", "woman", "boy", "girl", "lady", "gentleman", "stranger", "captain",
    "officer", "servant", "soldier", "fellow", "person", "doctor", "master", "sir",
})

_ZH_ALIAS_RIGHT_CONNECTOR_RE = re.compile(
    r"^[\s\u3000“”‘’\"'「」『』《》【】、：:之的叫为是作曰唤号名正就便乃即]*$"
)

_ALIAS_ZH_STRONG_CUES = (
    r"原名|本名|化名|改名为|改名叫|改名|改叫|改称|改姓|"
    r"真名叫|真名是|真名|道号|表字|法号|法名|"
    r"正是当年的|就是昔日的|原是当年的|便是当年的|正是昔日的|实乃|即是"
)

_ALIAS_ZH_CUE_RE = re.compile(
    rf"(?:{_ALIAS_ZH_STRONG_CUES})(?:叫做|叫作|叫|为|作|是|唤作|号|曰)?"
)

_ALIAS_EN_CUE_RE = re.compile(
    r"\b(?:formerly\s+known\s+as|also\s+known\s+as|under\s+the\s+name\s+of|"
    r"going\s+by\s+the\s+name\s+of|whose\s+real\s+name\s+(?:was|is)|"
    r"was\s+none\s+other\s+than|is\s+none\s+other\s+than|none\s+other\s+than)\b",
    re.IGNORECASE,
)

# 错别字“即是”当作“即使”（让步连词）守卫
_JISHI_CONCESSIVE_RE = re.compile(r"即是.{0,20}(?:不|也|都|还|亦|纵|仍|哪怕|就算|没|非|无)")

# 非同人关系硬否决守卫（作用于遮蔽双方人名后的连接片段与同句局部窗口）
_NON_SAME_RELATION_VETO_RE = re.compile(
    r"(?:"
    r"父亲|母亲|父子|母子|父女|母女|兄长|兄弟|姐妹|姊妹|妻子|丈夫|夫妻|夫妇|儿子|女儿|"
    r"之父|之母|之子|之女|之妻|之夫|之兄|之弟|之姐|之妹|之孙|之祖|之侄|之甥|之婿|"
    r"亲父|亲母|亲兄|亲弟|岳父|岳母|公公|婆婆|妻舅|郎舅|连襟|表哥|表弟|表姐|表妹|"
    r"伯父|叔父|舅父|姑母|姨母|嫂子|弟媳|女婿|儿媳|义父|义子|继父|继母|"
    r"师父|师尊|师傅|师兄|师弟|师姐|师妹|师叔|师伯|师祖|师徒|徒弟|弟子|门下|拜师|"
    r"至友|好友|挚友|朋友|结拜|义兄|义弟|同伙|伙伴|搭档|主仆|主子|仆人|奴才|属下|手下|上司|合伙|"
    r"宿敌|死敌|对头|仇人|冤家|追杀|击杀|击败|交手|决斗|交锋|斩杀|刺杀|讨伐|攻打|擒拿|捉拿|"
    r"杀|斩|刺|诛|仇|战|斗|"
    r"\b(?:father|mother|brother|sister|son|daughter|wife|husband|uncle|aunt|cousin|"
    r"nephew|niece|master|disciple|apprentice|student|teacher|friend|servant|companion|"
    r"enemy|rival|foe|killed|fought|slew|murdered|brother-in-law|father-in-law)\b"
    r")",
    re.IGNORECASE,
)

# C-sig2 两名之间出现典型跨主体互动动词／介词时否决（防“甲对改名为乙的丙说”误连甲与乙）
_BETWEEN_INTERACTION_VETO_RE = re.compile(
    r"(?:对|向|与|和|同|跟|给|让|令|派|请|邀|问|答|告诉|说道|道：|骂|打|看|望|见|遇|救|寻|找|带|领|领着|带着)"
)

# C-sig3 器物公开交战／夺宝／交易／赠予转移词
_OBJECT_TRANSFER_VETO_RE = re.compile(
    r"(?:夺|抢|抢夺|祭起|击|打|斩|战|杀|攻|赐|赠|送|借|卖|买|偷|盗|拾|捡|捡到|缴获|献|呈|"
    r"\b(?:seized|stole|stolen|gave|gifted|fought|captured|sold|bought|borrowed|handed)\b)",
    re.IGNORECASE,
)

# C-sig3 罕见标志性特征正则
_RARE_FEATURE_RE = re.compile(
    r"(?:胎记|断指|六指|缺指|独目|独眼|盲眼|瞎眼|瘸腿|跛脚|刺字|金印|刺青|纹身|"
    r"刀疤|剑疤|疤痕|伤疤|烙印|缺耳|兔唇|六趾|红痣|朱砂痣|黑痣|青记|赤记|"
    r"\b(?:birthmark|scar|missing\s+finger|one-eyed|tattoo|brand|limp|hunchback)\b)",
    re.IGNORECASE,
)

_SENT_SPLIT_RE = re.compile(r"[。！？；\n]+|(?<=[.!?])\s+")


def _normalize_text_key(s: str) -> str:
    return re.sub(
        r"[\s\u3000·•.,，、。！？；：\"'“”‘’()（）\[\]【】]+",
        "",
        str(s or "").strip(),
    ).lower()


def _is_valid_alias_surface(name: str) -> bool:
    s = str(name or "").strip()
    if len(s) < 2:
        return False
    if s.lower() in _GENERIC_PERSON_SURFACES or s in _GENERIC_PERSON_SURFACES:
        return False
    return True


def _entity_chapter_ords(
    entity: dict,
    mentions_by_eid: dict[str, list[dict]],
    body_seq: dict[str, int],
) -> list[int]:
    ords: set[int] = set()
    for m in mentions_by_eid.get(str(entity["id"]), []):
        cid = m.get("chapter_id")
        if cid is not None and str(cid) in body_seq:
            ords.add(body_seq[str(cid)])
    if not ords:
        for key in ("first_chapter_id", "last_chapter_id"):
            cid = entity.get(key)
            if cid is not None and str(cid) in body_seq:
                ords.add(body_seq[str(cid)])
    return sorted(ords)


def _chapter_overlap(ords_a: list[int], ords_b: list[int]) -> float:
    if not ords_a or not ords_b:
        return 0.0
    sa, sb = set(ords_a), set(ords_b)
    denom = min(len(sa), len(sb))
    if denom <= 0:
        return 0.0
    return round(len(sa & sb) / denom, 4)


def _chapter_span(ords: list[int]) -> list[int]:
    if not ords:
        return [0, 0]
    return [int(ords[0]), int(ords[-1])]


def _clip_verbatim(
    source_text: str,
    start: int,
    end: int,
    max_len: int = 160,
) -> tuple[int, int, str]:
    n = len(source_text)
    s = max(0, min(int(start), n))
    e = max(s, min(int(end), n))
    if e - s > max_len:
        e = s + max_len
    return s, e, source_text[s:e]


def _extract_masked_veto_hit(sent: str, name_a: str, name_b: str) -> bool:
    """将双方称呼本身遮蔽后检测是否存在亲属／师徒／朋友／敌对否决词。"""
    masked = sent
    for nm in sorted({name_a, name_b}, key=len, reverse=True):
        if nm:
            masked = masked.replace(nm, "□")
    return bool(_NON_SAME_RELATION_VETO_RE.search(masked))


def _is_english_word_boundary(text: str, start: int, end: int) -> bool:
    if re.search(r"[A-Za-z]", text[start:end]):
        if start > 0 and re.match(r"[A-Za-z0-9_]", text[start - 1]):
            return False
        if end < len(text) and re.match(r"[A-Za-z0-9_]", text[end]):
            return False
    return True


def propose_persona_link_hints(
    snap: dict | None,
    roster_out: dict | None,
    describe_doc: dict | None,
    source_text: str | None,
    chapters: list[dict] | None,
    *,
    mentions_override: list[dict] | None = None,
    top_n_anchor: int = 30,
    max_hints: int = 8,
) -> list[dict]:
    """融合三路确定性信号生成高信噪比跨马甲疑似同人待审提示（纯只读，不修改名册）。"""
    if not isinstance(snap, dict) or not isinstance(source_text, str) or not source_text:
        return []
    chapters_list = list(chapters or [])
    body_seq = _build_body_seq(chapters_list)
    raw_entities = list(snap.get("entities") or [])
    if mentions_override is not None:
        raw_mentions = list(mentions_override)
    else:
        raw_mentions = list((roster_out or {}).get("mentions") or snap.get("mentions") or [])

    mentions_by_eid: dict[str, list[dict]] = collections.defaultdict(list)
    for m in raw_mentions:
        eid = m.get("entity_id")
        if eid:
            mentions_by_eid[str(eid)].append(m)

    active_persons: list[dict] = []
    for ent in raw_entities:
        if ent.get("kind") != "person":
            continue
        status = ent.get("identity_status")
        if status in ("not_entity_confirmed", "not_entity_suggested"):
            continue
        eid = str(ent["id"])
        mcount = int(
            ent.get("mention_count")
            or len(mentions_by_eid.get(eid, []))
            or len(ent.get("mention_ids") or [])
        )
        ords = _entity_chapter_ords(ent, mentions_by_eid, body_seq)
        ch_cnt = max(len(ords), int(ent.get("chapter_count") or (1 if mcount > 0 else 0)))
        score = (
            float(ent.get("score"))
            if ent.get("score") is not None
            else math.sqrt(max(0, mcount) * max(0, ch_cnt))
        )
        active_persons.append({
            "id": eid,
            "name": str(ent.get("name") or eid),
            "names": list(dict.fromkeys([
                str(ent.get("name") or ""),
                *[str(x) for x in (ent.get("names") or [])],
                *[str(x) for x in (ent.get("aliases") or [])],
            ])),
            "mention_count": mcount,
            "chapter_ords": ords,
            "chapter_count": ch_cnt,
            "score": score,
            "raw_kind_rank": ent.get("kind_rank"),
        })

    if len(active_persons) < 2:
        return []

    active_persons.sort(
        key=lambda x: (
            int(x["raw_kind_rank"]) if x["raw_kind_rank"] is not None else 10**9,
            -x["score"],
            -x["mention_count"],
            -x["chapter_count"],
            x["id"],
        )
    )
    person_rank: dict[str, int] = {}
    person_by_id: dict[str, dict] = {}
    for idx, p in enumerate(active_persons, start=1):
        p["rank"] = idx
        person_rank[p["id"]] = idx
        person_by_id[p["id"]] = p

    anchor_ids = {p["id"] for p in active_persons[: max(1, int(top_n_anchor))]}
    pair_records: dict[tuple[str, str], dict] = {}

    def _register_signal(
        eid_a: str,
        eid_b: str,
        signal_name: str,
        ev_start: int | None,
        ev_end: int | None,
        ev_quote: str,
        detail: str = "",
    ) -> None:
        if eid_a == eid_b:
            return
        if eid_a not in person_by_id or eid_b not in person_by_id:
            return
        if eid_a not in anchor_ids and eid_b not in anchor_ids:
            return
        if person_rank[eid_a] <= person_rank[eid_b]:
            t_id, c_id = eid_a, eid_b
        else:
            t_id, c_id = eid_b, eid_a
        key = (t_id, c_id)
        rec = pair_records.get(key)
        if rec is None:
            rec = {
                "target_entity_id": t_id,
                "candidate_entity_id": c_id,
                "signals": [],
                "evidence_start": None,
                "evidence_end": None,
                "evidence_quote": "",
                "details": [],
            }
            pair_records[key] = rec
        if signal_name not in rec["signals"]:
            rec["signals"].append(signal_name)
        if detail:
            rec["details"].append(detail)
        if ev_start is not None and ev_end is not None and ev_quote:
            s_c, e_c, q_c = _clip_verbatim(source_text, ev_start, ev_end, 160)
            if q_c and source_text[s_c:e_c] == q_c:
                if not rec["evidence_quote"] or signal_name == "explicit_alias_text":
                    rec["evidence_start"] = s_c
                    rec["evidence_end"] = e_c
                    rec["evidence_quote"] = q_c

    # ------------------------------------------------------------------
    # 信号 1：名册 C2 同人提议／核验被局部共现拦截对
    # ------------------------------------------------------------------
    by_exact_name: dict[str, list[str]] = collections.defaultdict(list)
    for p in active_persons:
        for nm in p["names"]:
            if nm:
                by_exact_name[nm].append(p["id"])

    def _resolve_person_id_by_name(name: str, ev_addr: dict | None = None) -> str | None:
        if ev_addr and isinstance(ev_addr, dict) and ev_addr.get("start") is not None:
            es, ee = int(ev_addr["start"]), int(ev_addr["end"])
            for p in active_persons:
                for m in mentions_by_eid.get(p["id"], []):
                    if int(m["start"]) == es and int(m["end"]) == ee:
                        return p["id"]
        cands = by_exact_name.get(str(name or ""), [])
        if len(cands) == 1:
            return cands[0]
        if cands:
            return min(cands, key=lambda x: person_rank[x])
        return None

    if isinstance(roster_out, dict):
        for bp in roster_out.get("c2_blocked_pairs") or []:
            ea = str(bp.get("left_entity_id") or "")
            eb = str(bp.get("right_entity_id") or "")
            if not ea:
                ea = _resolve_person_id_by_name(bp.get("left_name", "")) or ""
            if not eb:
                eb = _resolve_person_id_by_name(bp.get("right_name", "")) or ""
            if ea and eb and ea != eb and ea in person_by_id and eb in person_by_id:
                es = bp.get("evidence_start")
                ee = bp.get("evidence_end")
                eq = ""
                if es is not None and ee is not None:
                    es, ee, eq = _clip_verbatim(source_text, int(es), int(ee), 160)
                elif mentions_by_eid.get(ea):
                    m0 = mentions_by_eid[ea][0]
                    lo = max(0, int(m0["start"]) - 20)
                    hi = min(len(source_text), int(m0["end"]) + 60)
                    es, ee, eq = _clip_verbatim(source_text, lo, hi, 160)
                _register_signal(ea, eb, "c2_blocked_same", es, ee, eq, "c2_blocked_pair")

        for rev in roster_out.get("reviews") or []:
            if rev.get("stage") != "roster-c2-verify":
                continue
            reason = str(rev.get("reason") or "")
            if "冲突" not in reason and "整组暂不合并" not in reason:
                continue
            member_names = list(rev.get("member_names") or [])
            if not member_names and rev.get("left_name") and rev.get("right_name"):
                member_names = [rev["left_name"], rev["right_name"]]
            ev_list = list(rev.get("evidence") or [])
            resolved_ids: list[str] = []
            for idx, nm in enumerate(member_names):
                addr = ev_list[idx] if idx < len(ev_list) else None
                pid = _resolve_person_id_by_name(str(nm), addr)
                if pid and pid not in resolved_ids:
                    resolved_ids.append(pid)
            for i in range(len(resolved_ids)):
                for j in range(i + 1, len(resolved_ids)):
                    ea, eb = resolved_ids[i], resolved_ids[j]
                    es, ee, eq = None, None, ""
                    if ev_list and isinstance(ev_list[0], dict) and ev_list[0].get("start") is not None:
                        s0 = max(0, int(ev_list[0]["start"]) - 20)
                        e0 = min(len(source_text), int(ev_list[0]["end"]) + 80)
                        es, ee, eq = _clip_verbatim(source_text, s0, e0, 160)
                    elif mentions_by_eid.get(ea):
                        m0 = mentions_by_eid[ea][0]
                        es, ee, eq = _clip_verbatim(
                            source_text,
                            max(0, int(m0["start"]) - 20),
                            int(m0["end"]) + 60,
                            160,
                        )
                    _register_signal(ea, eb, "c2_blocked_same", es, ee, eq, reason)

    # ------------------------------------------------------------------
    # 信号 2：原文同句紧邻显式本名／化名／旧称等同揭示（带硬否决守卫）
    # ------------------------------------------------------------------
    surface_to_eids: dict[str, set[str]] = collections.defaultdict(set)
    for p in active_persons:
        if p["mention_count"] < 2 and p["id"] not in anchor_ids:
            continue
        cand_surfs = [p["name"]]
        for nm in p["names"]:
            if nm != p["name"] and _is_valid_alias_surface(nm) and len(nm) >= 2:
                cand_surfs.append(nm)
        for nm in cand_surfs[:4]:
            if _is_valid_alias_surface(nm):
                surface_to_eids[nm].add(p["id"])

    unique_surfaces: list[tuple[str, str]] = [
        (surf, next(iter(eids)))
        for surf, eids in surface_to_eids.items()
        if len(eids) == 1
    ]
    unique_surfaces.sort(key=lambda x: (-len(x[0]), x[0]))

    if unique_surfaces:
        para_pos = 0
        for raw_para in source_text.split("\n"):
            para_start = para_pos
            para_pos += len(raw_para) + 1
            if not raw_para.strip():
                continue
            if not _ALIAS_ZH_CUE_RE.search(raw_para) and not _ALIAS_EN_CUE_RE.search(raw_para):
                continue

            sent_cursor = 0
            for sent in _SENT_SPLIT_RE.split(raw_para):
                if not sent:
                    continue
                s_off = raw_para.find(sent, sent_cursor)
                if s_off < 0:
                    continue
                sent_cursor = s_off + len(sent)
                s_zh = list(_ALIAS_ZH_CUE_RE.finditer(sent))
                s_en = list(_ALIAS_EN_CUE_RE.finditer(sent))
                if not s_zh and not s_en:
                    continue

                occ: list[tuple[int, int, str, str]] = []
                occupied: list[tuple[int, int]] = []
                for surf, eid in unique_surfaces:
                    if surf not in sent:
                        continue
                    search_from = 0
                    while True:
                        idx = sent.find(surf, search_from)
                        if idx < 0:
                            break
                        search_from = idx + len(surf)
                        end_idx = idx + len(surf)
                        if not _is_english_word_boundary(sent, idx, end_idx):
                            continue
                        if any(not (end_idx <= os_ or idx >= oe_) for os_, oe_ in occupied):
                            continue
                        occupied.append((idx, end_idx))
                        occ.append((idx, end_idx, surf, eid))
                if len(occ) < 2:
                    continue
                occ.sort(key=lambda x: (x[0], x[1]))

                all_cues = [(m.start(), m.end(), m.group(0), "zh") for m in s_zh] + [
                    (m.start(), m.end(), m.group(0), "en") for m in s_en
                ]
                for cs, ce, ctxt, lang in all_cues:
                    if lang == "zh" and ctxt.startswith("即是"):
                        if _JISHI_CONCESSIVE_RE.search(sent[max(0, cs - 4):min(len(sent), ce + 24)]):
                            continue
                    left_occ = [o for o in occ if o[1] <= cs]
                    right_occ = [o for o in occ if o[0] >= ce]
                    if left_occ and right_occ:
                        oa = left_occ[-1]
                        ob = right_occ[0]
                        if oa[3] != ob[3] and oa[2] not in ob[2] and ob[2] not in oa[2]:
                            gap_l = cs - oa[1]
                            gap_r = ob[0] - ce
                            dist = ob[0] - oa[1]
                            max_l = 22 if lang == "zh" else 35
                            max_r = 14 if lang == "zh" else 30
                            if gap_l <= max_l and gap_r <= max_r and dist <= 300:
                                between_l = sent[oa[1]:cs]
                                between_r = sent[ce:ob[0]]
                                span_lo = max(0, oa[0] - 8)
                                span_hi = min(len(sent), ob[1] + 8)
                                local_slice = sent[span_lo:span_hi]
                                tight_ok = True
                                if lang == "zh" and ctxt.startswith(("实乃", "即是")):
                                    if gap_l > 8 or gap_r > 8 or re.search(r"[，、：]", between_l + between_r):
                                        tight_ok = False
                                if tight_ok and lang == "zh" and _BETWEEN_INTERACTION_VETO_RE.search(between_l):
                                    tight_ok = False
                                if tight_ok and lang == "zh" and not _ZH_ALIAS_RIGHT_CONNECTOR_RE.match(between_r):
                                    tight_ok = False
                                if tight_ok and not _extract_masked_veto_hit(local_slice, oa[2], ob[2]):
                                    abs_s = para_start + s_off + span_lo
                                    abs_e = para_start + s_off + span_hi
                                    _register_signal(
                                        oa[3], ob[3], "explicit_alias_text",
                                        abs_s, abs_e, source_text[abs_s:abs_e],
                                        f"forward:{ctxt}",
                                    )

                    if lang == "zh" and len(right_occ) >= 2:
                        o1, o2 = right_occ[0], right_occ[1]
                        if o1[3] != o2[3] and o1[2] not in o2[2] and o2[2] not in o1[2]:
                            gap_r1 = o1[0] - ce
                            mid_12 = sent[o1[1]:o2[0]]
                            if gap_r1 <= 6 and len(mid_12) <= 6 and ("的" in mid_12 or "之" in mid_12):
                                span_lo = max(0, cs - 4)
                                span_hi = min(len(sent), o2[1] + 6)
                                local_slice = sent[span_lo:span_hi]
                                if not _extract_masked_veto_hit(local_slice, o1[2], o2[2]):
                                    abs_s = para_start + s_off + span_lo
                                    abs_e = para_start + s_off + span_hi
                                    _register_signal(
                                        o1[3], o2[3], "explicit_alias_text",
                                        abs_s, abs_e, source_text[abs_s:abs_e],
                                        f"modifier:{ctxt}",
                                    )

    # ------------------------------------------------------------------
    # 信号 3：前后时期互补出场 + 共享非易主专名持有物／罕见标志性特征
    # ------------------------------------------------------------------
    if isinstance(describe_doc, dict):
        for ent in describe_doc.get("entities") or []:
            if ent.get("kind") != "object":
                continue
            obj_name = str(ent.get("name") or "")
            if len(obj_name) < 2:
                continue
            if any(ev.get("event_type") == "owner_change" for ev in (ent.get("state_events") or [])):
                continue
            holder_facts: dict[str, list[dict]] = collections.defaultdict(list)
            for f in ent.get("facts") or []:
                if f.get("facet") != "holder":
                    continue
                hid = f.get("holder_entity_id")
                if not hid or str(hid) not in person_by_id:
                    continue
                qtxt = str(f.get("quote") or "")
                vtxt = str(f.get("value") or "")
                if _OBJECT_TRANSFER_VETO_RE.search(qtxt) or _OBJECT_TRANSFER_VETO_RE.search(vtxt):
                    continue
                holder_facts[str(hid)].append(f)
            h_ids = sorted(holder_facts)
            for i in range(len(h_ids)):
                for j in range(i + 1, len(h_ids)):
                    ea, eb = h_ids[i], h_ids[j]
                    pa, pb = person_by_id[ea], person_by_id[eb]
                    if len(pa["chapter_ords"]) < 2 or len(pb["chapter_ords"]) < 2:
                        continue
                    ov = _chapter_overlap(pa["chapter_ords"], pb["chapter_ords"])
                    if ov > 0.15:
                        continue
                    med_a = pa["chapter_ords"][len(pa["chapter_ords"]) // 2]
                    med_b = pb["chapter_ords"][len(pb["chapter_ords"]) // 2]
                    if med_a == med_b:
                        continue
                    early_p, late_p = (pa, pb) if med_a < med_b else (pb, pa)
                    early_before = sum(
                        1 for c in early_p["chapter_ords"] if c <= late_p["chapter_ords"][0] + 2
                    )
                    if early_before / len(early_p["chapter_ords"]) < 0.70:
                        continue
                    f0 = holder_facts[ea][0]
                    _register_signal(
                        ea, eb, "complementary_chapters_shared_anchor",
                        int(f0["start"]), int(f0["end"]), str(f0["quote"]),
                        f"shared_object:{obj_name}",
                    )

        rare_feat_holders: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
        for ent in describe_doc.get("entities") or []:
            if ent.get("kind") != "person":
                continue
            eid = str(ent.get("entity_id") or ent.get("id") or "")
            if eid not in person_by_id:
                continue
            seen_keys: set[str] = set()
            for f in ent.get("facts") or []:
                if f.get("facet") != "feature":
                    continue
                val = str(f.get("value") or "")
                quo = str(f.get("quote") or "")
                if not (_RARE_FEATURE_RE.search(val) or _RARE_FEATURE_RE.search(quo)):
                    continue
                nk = _normalize_text_key(val)
                if len(nk) < 2 or nk in seen_keys:
                    continue
                seen_keys.add(nk)
                rare_feat_holders[nk].append((eid, f))

        for nk, holders in rare_feat_holders.items():
            uniq_eids = list(dict.fromkeys(h[0] for h in holders))
            if len(uniq_eids) != 2:
                continue
            ea, eb = uniq_eids[0], uniq_eids[1]
            pa, pb = person_by_id[ea], person_by_id[eb]
            if len(pa["chapter_ords"]) < 2 or len(pb["chapter_ords"]) < 2:
                continue
            ov = _chapter_overlap(pa["chapter_ords"], pb["chapter_ords"])
            if ov > 0.15:
                continue
            f0 = holders[0][1]
            _register_signal(
                ea, eb, "complementary_chapters_shared_anchor",
                int(f0["start"]), int(f0["end"]), str(f0["quote"]),
                f"rare_feature:{nk}",
            )

    # ------------------------------------------------------------------
    # 汇总、定级、排序与截断
    # ------------------------------------------------------------------
    hints: list[dict] = []
    for (t_id, c_id), rec in pair_records.items():
        pt, pc = person_by_id[t_id], person_by_id[c_id]
        ov = _chapter_overlap(pt["chapter_ords"], pc["chapter_ords"])
        sigs = list(rec["signals"])
        if (
            len(sigs) >= 2
            or "c2_blocked_same" in sigs
            or ("explicit_alias_text" in sigs and ov <= 0.25)
        ):
            tier = "high"
        else:
            tier = "medium"

        ev_start = rec["evidence_start"]
        ev_end = rec["evidence_end"]
        ev_quote = rec["evidence_quote"]
        if ev_start is not None and ev_end is not None:
            if not (0 <= ev_start <= ev_end <= len(source_text) and source_text[ev_start:ev_end] == ev_quote):
                ev_start, ev_end, ev_quote = None, None, ""

        hid = digest(["persona-hint-v1", t_id, c_id, sorted(sigs)])
        hints.append({
            "hint_id": hid,
            "target_entity_id": t_id,
            "target_name": pt["name"],
            "candidate_entity_id": c_id,
            "candidate_name": pc["name"],
            "signals": sigs,
            "confidence_tier": tier,
            "target_chapter_span": _chapter_span(pt["chapter_ords"]),
            "candidate_chapter_span": _chapter_span(pc["chapter_ords"]),
            "chapter_overlap_ratio": ov,
            "evidence_quote": ev_quote,
            "evidence_start": ev_start,
            "evidence_end": ev_end,
            "suggested_cli": (
                f"entitytrace correct <run_id> merge {t_id} {c_id} "
                f"--source user --expect-revision <rev>"
            ),
            "_sort_key": (
                0 if tier == "high" else 1,
                -len(sigs),
                ov,
                pt["rank"],
                pc["rank"],
                hid,
            ),
        })

    hints.sort(key=lambda x: x["_sort_key"])
    result: list[dict] = []
    for h in hints[: max(0, int(max_hints))]:
        item = dict(h)
        item.pop("_sort_key", None)
        result.append(item)
    return result
