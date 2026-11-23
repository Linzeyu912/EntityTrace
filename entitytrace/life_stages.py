"""人物跨时期人生阶段确定性切分模块（纯函数、零 IO、零网络请求）。

基于人物已核验 `age` 事实序列解析凡人年龄区间 `[low, high]` 与五级视觉人生阶段桶
（`child / teen / young_adult / middle / senior`），叠加四道确定性守卫：
1. 相对比较、非年龄时长与倒叙回忆剔除；
2. 显式易容伪装剔除（不泛杀正常外貌旁观措辞）；
3. 跨桶最小年龄差、跨章推进与全局后缀单调不回退守卫；
4. 既有事件切点去重与主导长断档切界对齐（触发事件始终保留逐字原文地址）。
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .util import digest

LIFE_STAGE_BUCKETS: tuple[str, ...] = (
    "child",
    "teen",
    "young_adult",
    "middle",
    "senior",
)

LIFE_STAGE_BUCKET_LABELS: dict[str, str] = {
    "child": "幼童期",
    "teen": "少年期",
    "young_adult": "青年期",
    "middle": "中年期",
    "senior": "老年期",
}

TRIGGER_SOURCE_LIFE_STAGE = "life_stage_age_sequence"

_MORTAL_LOW_MAX = 120.0
_MORTAL_HIGH_MAX = 150.0
_SUPERNATURAL = object()


@dataclass(frozen=True)
class AgeSignal:
    low: float
    high: float
    bucket_low: int
    bucket_high: int
    is_vague: bool
    parse_kind: str


def age_to_bucket(age: float) -> int:
    """将凡人数值年龄映射到五级通用视觉人生阶段桶索引（0..4）。"""
    val = float(age)
    if val <= 12.0:
        return 0
    if val < 18.0:
        return 1
    if val < 30.0:
        return 2
    if val < 55.0:
        return 3
    return 4


def _make_signal(
    low: float,
    high: float,
    *,
    is_vague: bool = False,
    parse_kind: str = "numeric",
) -> AgeSignal | None:
    lo = float(min(low, high))
    hi = float(max(low, high))
    if lo < 0.0 or lo > _MORTAL_LOW_MAX or hi > _MORTAL_HIGH_MAX:
        return None
    return AgeSignal(
        low=lo,
        high=hi,
        bucket_low=age_to_bucket(lo),
        bucket_high=age_to_bucket(hi),
        is_vague=bool(is_vague),
        parse_kind=parse_kind,
    )


# ---------------------------------------------------------------------------
# 守卫 1：相对比较、非年龄存续时长与倒叙回忆剔除
# ---------------------------------------------------------------------------

_RELATIVE_OR_DURATION_RE = re.compile(
    r"(?:"
    r"比.{1,10}(?:小|大|年轻|年长)"
    r"|相差.{1,6}岁"
    r"|最年幼|最年长|姐妹中最|排行老"
    r"|\b(?:junior|senior|eldest|youngest|older\s+than|younger\s+than)\b"
    r"|\babided\s+for\s+.+?\byears?\b"
    r"|\bfor\s+now\s+some\s+.+?\byears?\b"
    r")",
    re.IGNORECASE,
)

_FLASHBACK_VALUE_END_RE = re.compile(
    r"(?:时|之时|时候|那年|当年|以前|之前)[。，、；：！？\"'”’）)\s]*$"
)

_FLASHBACK_VALUE_RE = re.compile(
    r"(?:"
    r"幼时|小时候|童年时|少时|少年时|年轻时|青年时|中年时|出生时|襁褓中"
    r"|当年|昔年|昔日|往昔|早年|从前|曾经|多年前|年前"
    r"|未[\u4e00-\u9fff]{1,8}时|初见时|转.{1,6}后又过"
    r")"
)

_FLASHBACK_QUOTE_EN_RE = re.compile(
    r"\b(?:"
    r"when\s+\w+\s+was"
    r"|was\s+then"
    r"|years?\s+ago"
    r"|in\s+(?:his|her|their|my)\s+youth"
    r"|used\s+to\s+be"
    r"|had\s+been"
    r"|not\s+till\s+\w+\s+was"
    r"|back\s+when"
    r")\b",
    re.IGNORECASE,
)

_FLASHBACK_QUOTE_ZH_RE = re.compile(
    r"(?:"
    r"[一二两三四五六七八九十\d]+岁(?:那年|之时|时|那会儿|的时候)"
    r"|方被"
    r"|当年|昔年|昔日|往昔|幼时|小时候|童年时|少时|少年时|年轻时|青年时|中年时"
    r"|早年|从前|曾经|多年前|年前|未[\u4e00-\u9fff]{1,8}时|初见时"
    r")"
)

_PRESENT_ANCHOR_ZH_RE = re.compile(
    r"(?:如今|今年|现在|目今|此时|已经|已满|正值|眼下|目前)"
)

# ---------------------------------------------------------------------------
# 守卫 2：显式易容伪装剔除（严禁泛杀“看起来”“模样”“样子”）
# ---------------------------------------------------------------------------

_DISGUISE_RE = re.compile(
    r"(?:"
    r"易容|伪装|假扮|乔装|掩饰"
    r"|并非.{0,4}真正容貌|不是.{0,8}真正容貌"
    r"|假相|幻化为|变幻成"
    r"|\bdisguis(?:e|ed)\b|\bin\s+disguise\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 数值解析辅助（阿拉伯数字、中文数字／连数／旬／雅称、英文数字）
# ---------------------------------------------------------------------------

_ZH_DIGIT: dict[str, int] = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

_NON_AGE_UNIT_AFTER = "月日号层重个名位尺寸斤两次回章卷步丈里倍分天刻点"

# 明确超凡／百千岁模式（排除“半百”）
_SUPERNATURAL_ZH_RE = re.compile(
    r"(?<!半)(?:"
    r"[一二两三四五六七八九]{1,2}[百千万][余多几来许]?(?:岁|年|载)"
    r"|[千万][余多几来许]?(?:岁|年|载)"
    r"|一百[二三四五六七八九]十[一二三四五六七八九]?(?:余|多)?岁"
    r")"
)

# 古典雅称（含精确／区间年龄语义，优先于一般中文数字和定性词）
_CLASSICAL_IDIOM_SPECS: tuple[tuple[str, float, float], ...] = (
    ("垂髫", 3.0, 8.0),
    ("总角", 8.0, 12.0),
    ("豆蔻", 13.0, 14.0),
    ("及笄", 15.0, 15.0),
    ("弱冠", 20.0, 20.0),
    ("而立", 30.0, 30.0),
    ("不惑", 40.0, 40.0),
    ("知天命", 50.0, 50.0),
    ("知命", 50.0, 50.0),
    ("年过半百", 51.0, 54.0),
    ("半百", 50.0, 50.0),
    ("花甲", 60.0, 60.0),
    ("古稀", 70.0, 70.0),
    ("耄耋", 80.0, 90.0),
)

# 阿拉伯数字模式
_ARABIC_RANGE_RE = re.compile(
    r"(?<!\d)(\d{1,4})\s*(?:[—–\-~～至到或/]|to|or)\s*(\d{1,4})"
    r"\s*(多|余|来|许)?\s*(岁|周岁|虚岁|年纪|年岁|years?\s*old|yo)?",
    re.IGNORECASE,
)

_ARABIC_SINGLE_RE = re.compile(
    r"(年近|将近|快|不到|不满|不超过|不足|未满|刚过|刚满|年仅|年方|年约|年及|年过|年逾|大约|约|才|已满|虚度|亡年|过)?"
    r"\s*(?<!\d)(\d{1,4})(?!\d)\s*"
    r"(多|有余|余|许|来|几|开外|出头|以上|之上|左右|上下)?"
    r"\s*(岁|周岁|虚岁|年纪|年岁|years?\s*(?:old|of\s+age)|yo)?"
    r"(多|有余|余|许|来|开外|出头|以上|之上|左右|上下)?",
    re.IGNORECASE,
)

# 中文特殊约数：十来岁 / 十几岁
_ZH_SHILAI_RE = re.compile(r"(?<![一二两三四五六七八九十])十来(?:岁|周岁|虚岁)?")
_ZH_SHIJI_RE = re.compile(r"(?<![一二两三四五六七八九十])十(?:几|余|多|数)(?:岁|周岁|虚岁)?")

# 中文连数：二十四五岁 / 十四五岁 / 三四十岁 / 五六岁
_ZH_TENS_CONSEC_RE = re.compile(
    r"([二三四五六七八九])十([一二三四五六七八])([二三四五六七八九])(?:岁|周岁|虚岁|年纪|年岁)?"
)
_ZH_TEEN_CONSEC_RE = re.compile(
    r"(?<![一二两三四五六七八九十])十([一二三四五六七八])([二三四五六七八九])(?:岁|周岁|虚岁|年纪|年岁)?"
)
_ZH_DECADE_CONSEC_RE = re.compile(
    r"(?<![一二两三四五六七八九十])([一二两三四五六七八])([二三四五六七八九])十"
    r"(多|余|来|许)?(?:岁|周岁|虚岁|年纪|年岁)?"
)
_ZH_SINGLE_CONSEC_RE = re.compile(
    r"(?<![一二两三四五六七八九十])([一二两三四五六七八])([二两三四五六七八九])(?:岁|周岁|虚岁)"
)

# 中文显式“或／至／到／、”范围：约十五或十六岁 / 二十至二十五岁 / 十五、六岁
_ZH_NUM_ATOM = r"(?:[一二两三四五六七八九]?十[一二三四五六七八九]?|[一二两三四五六七八九])"
_ZH_EXPLICIT_RANGE_RE = re.compile(
    rf"({_ZH_NUM_ATOM})\s*(?:或|至|到|—|–|-|~|～|、)\s*({_ZH_NUM_ATOM})\s*(?:岁|周岁|虚岁|年纪|年岁)"
)

# 古典“二十有五”
_ZH_YOU_RE = re.compile(r"([二三四五六七八九])十有([一二三四五六七八九])")

# 旬（排除“中旬”）
_ZH_XUN_RE = re.compile(
    r"(年近|将近|年约|年及|年过|年逾|过|约|近)?\s*([二三四五六七八九])旬(以上|之上|开外|出头|有余|余|多|左右|上下)?"
)

# 标准中文数字年龄：支持前缀修饰、后缀修饰
_ZH_SINGLE_RE = re.compile(
    r"(年近|将近|快|不到|不满|不超过|不足|未满|刚过|刚满|年仅|年方|年约|年及|年过|年逾|大约|约|才|已满|虚度|亡年|过)?"
    r"([一二两三四五六七八九]?十[一二三四五六七八九]?|[一二两三四五六七八九])"
    r"(多|有余|余|许|来|几|开外|出头|以上|之上|左右|上下)?"
    r"(岁|周岁|虚岁|年纪|年岁)?"
    r"(多|有余|余|许|来|开外|出头|以上|之上|左右|上下)?"
)

# 英文数字解析
_EN_ONES: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_EN_TEENS: dict[str, int] = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_DECADES: dict[str, tuple[float, float]] = {
    "teens": (13.0, 19.0),
    "twenties": (20.0, 29.0),
    "thirties": (30.0, 39.0),
    "forties": (40.0, 49.0),
    "fifties": (50.0, 59.0),
    "sixties": (60.0, 69.0),
    "seventies": (70.0, 79.0),
    "eighties": (80.0, 89.0),
    "nineties": (90.0, 99.0),
}

_EN_DECADE_PHRASE_RE = re.compile(
    r"\bin\s+(?:his|her|their)\s+(?:(early|mid|late)[-\s]+)?"
    r"(teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
    re.IGNORECASE,
)

_EN_ONES_PAT = "|".join(_EN_ONES)
_EN_TEENS_PAT = "|".join(_EN_TEENS)
_EN_TENS_PAT = "|".join(_EN_TENS)
_EN_NUM_WORD_PAT = (
    rf"(?:(?:{_EN_ONES_PAT})[-\s]+and[-\s]+(?:{_EN_TENS_PAT})"
    rf"|(?:{_EN_TENS_PAT})(?:[-\s]+(?:{_EN_ONES_PAT}))?"
    rf"|(?:{_EN_TEENS_PAT})"
    rf"|(?:{_EN_ONES_PAT}))"
)

_EN_RANGE_RE = re.compile(
    rf"\b({_EN_NUM_WORD_PAT})\s+(?:or|to)\s+({_EN_NUM_WORD_PAT})\b",
    re.IGNORECASE,
)
_EN_MOD_NUM_RE = re.compile(
    rf"(?:\b(almost|nearly|under|less\s+than|not\s+more\s+than|cannot\s+be\s+more\s+than|"
    rf"little\s+more\s+than|just\s+over|over|about|around|not|only|aged?|at\s+the\s+age\s+of|at|of)\s+)?"
    rf"\b({_EN_NUM_WORD_PAT})\b"
    rf"(?:\s+(years?\s*(?:old|of\s+age)|or\s+more))?",
    re.IGNORECASE,
)

# 英文引用场景安全锚词（防将 "three months" 等误识为年龄）
_EN_AGE_CUE_RE = re.compile(
    r"\b(?:years?\s*(?:old|of\s+age)|aged?|age\s+of|girl\s+of|boy\s+of|man\s+of|woman\s+of|"
    r"almost|nearly|about|around|under|little\s+more\s+than|more\s+than|married\s+at|only\s+\w+\s+last|"
    r"in\s+(?:his|her|their)\s+(?:early|mid|late)?\s*"
    r"(?:teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties))\b",
    re.IGNORECASE,
)
_EN_NON_AGE_FOLLOW_RE = re.compile(
    r"^\s*(?:months?|weeks?|days?|hours?|minutes?|times?|miles?|feet|inches|"
    r"brothers?|sisters?|men|women|people|ships?|pounds?|guineas|francs|dollars)\b",
    re.IGNORECASE,
)


def _zh_to_int(token: str) -> int | None:
    s = token.strip()
    if not s:
        return None
    if s in _ZH_DIGIT:
        return _ZH_DIGIT[s]
    if s == "十":
        return 10
    if "十" in s:
        parts = s.split("十", 1)
        tens = 1 if not parts[0] else _ZH_DIGIT.get(parts[0])
        ones = 0 if not parts[1] else _ZH_DIGIT.get(parts[1])
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return None


def _en_to_int(token: str) -> int | None:
    s = re.sub(r"[-\s]+", " ", token.strip().lower())
    if s in _EN_ONES:
        return _EN_ONES[s]
    if s in _EN_TEENS:
        return _EN_TEENS[s]
    if s in _EN_TENS:
        return _EN_TENS[s]
    m_inv = re.fullmatch(rf"({_EN_ONES_PAT}) and ({_EN_TENS_PAT})", s)
    if m_inv:
        return _EN_TENS[m_inv.group(2)] + _EN_ONES[m_inv.group(1)]
    m_std = re.fullmatch(rf"({_EN_TENS_PAT}) ({_EN_ONES_PAT})", s)
    if m_std:
        return _EN_TENS[m_std.group(1)] + _EN_ONES[m_std.group(2)]
    return None


def _apply_zh_modifiers(
    base: float,
    prefix: str | None,
    suffix: str | None,
) -> tuple[float, float]:
    pre = (prefix or "").strip()
    suf = (suffix or "").strip()
    if suf in ("多", "有余", "余", "许", "几"):
        if base == 10.0:
            return (13.0, 17.0)
        if base >= 20.0 and int(base) % 10 == 0:
            return (base + 1.0, base + 9.0)
        return (base, base + 2.0)
    if suf == "来":
        if base == 10.0:
            return (10.0, 12.0)
        if base >= 20.0 and int(base) % 10 == 0:
            return (base, base + 4.0 if base == 50.0 else base + 5.0)
        return (max(0.0, base - 1.0), base + 1.0)
    if suf == "开外":
        if base >= 20.0 and int(base) % 10 == 0:
            return (base + 1.0, base + 9.0)
        return (base + 1.0, base + 5.0)
    if suf == "出头":
        return (base, base + 3.0)
    if suf in ("以上", "之上"):
        return (base, base + 5.0)
    if pre in ("不到", "不满", "不足", "未满"):
        return (max(0.0, base - 2.0), max(0.0, base - 0.5))
    if pre == "不超过":
        return (max(0.0, base - 2.0), base)
    if pre in ("年近", "将近", "快"):
        return (max(0.0, base - 2.0), max(0.0, base - 0.1))
    if pre in ("年过", "年逾"):
        return (base + 0.1, base + 4.0 if base == 50.0 else base + 5.0)
    if pre in ("刚过", "过"):
        return (base + 0.1, base + 1.0)
    return (base, base)


def _parse_numeric_text(text: str, *, strict_unit: bool) -> AgeSignal | None | object:
    """尝试从单个字符串解析数值年龄；遇到超凡高龄返回 `_SUPERNATURAL`。"""
    s = (text or "").strip()
    if not s:
        return None

    # 1. 古典雅称（放在超凡检测前，保证“半百”不被误判为百岁以上）
    for idiom, lo, hi in _CLASSICAL_IDIOM_SPECS:
        if idiom in s:
            return _make_signal(lo, hi, is_vague=False, parse_kind="classical_idiom")

    # 2. 中文超凡高龄拦截（如“六百余岁”“四五百年”）
    if _SUPERNATURAL_ZH_RE.search(s):
        return _SUPERNATURAL

    # 3. 阿拉伯数字区间（如 25—26岁、15-16）
    for m in _ARABIC_RANGE_RE.finditer(s):
        a, b = float(m.group(1)), float(m.group(2))
        unit = m.group(4) or ""
        tail = s[m.end():m.end() + 1]
        if tail and tail in _NON_AGE_UNIT_AFTER:
            continue
        if strict_unit and not unit:
            continue
        if a > _MORTAL_LOW_MAX or b > _MORTAL_HIGH_MAX:
            return _SUPERNATURAL
        if b < a:
            continue
        sig = _make_signal(a, b, is_vague=False, parse_kind="arabic_range")
        if sig is not None:
            return sig

    # 4. 阿拉伯数字单值／约数／修饰词
    for m in _ARABIC_SINGLE_RE.finditer(s):
        prefix, num_str, suf1, unit, suf2 = (
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        )
        suffix = suf1 or suf2
        tail = s[m.end():m.end() + 1]
        if tail and tail in _NON_AGE_UNIT_AFTER:
            continue
        has_cue = bool(prefix or unit or suffix in ("开外", "出头"))
        if strict_unit and not (unit or (prefix and prefix in ("年近", "年仅", "年方", "年约", "年及", "年过", "年逾", "虚度", "亡年"))):
            continue
        if not strict_unit and not has_cue and not re.fullmatch(r"\d{1,4}", s):
            continue
        val = float(num_str)
        if val > _MORTAL_LOW_MAX:
            return _SUPERNATURAL
        lo, hi = _apply_zh_modifiers(val, prefix, suffix)
        if lo > _MORTAL_LOW_MAX or hi > _MORTAL_HIGH_MAX:
            return _SUPERNATURAL
        sig = _make_signal(lo, hi, is_vague=False, parse_kind="arabic_num")
        if sig is not None:
            return sig

    # 5. 中文特例：“十来岁” -> [10, 12]，“十几岁／十余岁” -> [13, 17]
    m_sl = _ZH_SHILAI_RE.search(s)
    if m_sl:
        if not strict_unit or "岁" in m_sl.group(0):
            return _make_signal(10.0, 12.0, is_vague=False, parse_kind="zh_approx")
    m_sj = _ZH_SHIJI_RE.search(s)
    if m_sj:
        if not strict_unit or "岁" in m_sj.group(0):
            return _make_signal(13.0, 17.0, is_vague=False, parse_kind="zh_approx")

    # 6. 中文连数：二十四五岁、十四五岁、三四十岁、五六岁
    for m in _ZH_TENS_CONSEC_RE.finditer(s):
        tens = _ZH_DIGIT[m.group(1)]
        d1, d2 = _ZH_DIGIT[m.group(2)], _ZH_DIGIT[m.group(3)]
        if d2 > d1:
            if strict_unit and not any(k in m.group(0) for k in ("岁", "年纪", "年岁")) and "年纪" not in s[:m.start()]:
                continue
            return _make_signal(tens * 10 + d1, tens * 10 + d2, is_vague=False, parse_kind="zh_consec")

    for m in _ZH_TEEN_CONSEC_RE.finditer(s):
        d1, d2 = _ZH_DIGIT[m.group(1)], _ZH_DIGIT[m.group(2)]
        if d2 > d1:
            if strict_unit and not any(k in m.group(0) for k in ("岁", "年纪", "年岁")) and "年纪" not in s[:m.start()]:
                continue
            return _make_signal(10 + d1, 10 + d2, is_vague=False, parse_kind="zh_consec")

    for m in _ZH_DECADE_CONSEC_RE.finditer(s):
        d1, d2 = _ZH_DIGIT[m.group(1)], _ZH_DIGIT[m.group(2)]
        if d2 > d1:
            if strict_unit and not any(k in m.group(0) for k in ("岁", "年纪", "年岁")):
                continue
            tail = s[m.end():m.end() + 1]
            if tail and tail in _NON_AGE_UNIT_AFTER:
                continue
            return _make_signal(d1 * 10.0, d2 * 10.0, is_vague=False, parse_kind="zh_consec")

    for m in _ZH_SINGLE_CONSEC_RE.finditer(s):
        d1, d2 = _ZH_DIGIT[m.group(1)], _ZH_DIGIT[m.group(2)]
        if d2 > d1:
            return _make_signal(float(d1), float(d2), is_vague=False, parse_kind="zh_consec")

    # 7. 中文显式范围：约十五或十六岁、十五、六岁
    for m in _ZH_EXPLICIT_RANGE_RE.finditer(s):
        a = _zh_to_int(m.group(1))
        b = _zh_to_int(m.group(2))
        if a is not None and b is not None:
            # 处理“十五、六岁”这类省略十位的写法
            if a >= 10 and b < 10 and b > (a % 10):
                b = (a // 10) * 10 + b
            if b >= a:
                sig = _make_signal(float(a), float(b), is_vague=False, parse_kind="zh_range")
                if sig is not None:
                    return sig

    # 8. 古典“二十有五”
    m_you = _ZH_YOU_RE.search(s)
    if m_you:
        val = _ZH_DIGIT[m_you.group(1)] * 10 + _ZH_DIGIT[m_you.group(2)]
        return _make_signal(float(val), float(val), is_vague=False, parse_kind="zh_you")

    # 9. “旬”（三旬／四旬／五旬／六旬／七旬／八旬）
    m_xun = _ZH_XUN_RE.search(s)
    if m_xun:
        prefix = m_xun.group(1)
        base = float(_ZH_DIGIT[m_xun.group(2)] * 10)
        suf = m_xun.group(3)
        if suf in ("以上", "之上", "开外", "有余", "余", "多"):
            lo, hi = (base + 1.0, base + 9.0) if base == 50.0 else (base, base + 5.0)
        elif suf == "出头":
            lo, hi = (base, base + 3.0)
        elif prefix in ("年近", "将近", "近"):
            lo, hi = (base - 2.0, base - 0.1)
        elif prefix in ("年过", "年逾", "过"):
            lo, hi = (base + 0.1, base + 4.0 if base == 50.0 else base + 5.0)
        else:
            lo, hi = (base, base)
        sig = _make_signal(lo, hi, is_vague=False, parse_kind="zh_xun")
        if sig is not None:
            return sig

    # 10. 标准中文数字（优先选择带明确年龄词/岁字的匹配，跳过月份等非年龄单位）
    best_zh: tuple[int, float, float] | None = None
    for m in _ZH_SINGLE_RE.finditer(s):
        prefix, num_tok, suf1, unit, suf2 = (
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        )
        suffix = suf1 or suf2
        tail = s[m.end():m.end() + 1]
        if tail and tail in _NON_AGE_UNIT_AFTER:
            continue
        val_int = _zh_to_int(num_tok)
        if val_int is None:
            continue
        has_strong_prefix = prefix in ("年近", "将近", "年仅", "年方", "年约", "年及", "年过", "年逾", "虚度", "亡年", "刚满", "刚过", "不到", "不满", "不超过", "不足", "未满")
        has_unit = bool(unit)
        if strict_unit and not (has_unit or prefix in ("年近", "年仅", "年方", "年约", "年及", "年过", "年逾", "虚度", "亡年")):
            continue
        if not strict_unit and not (has_unit or has_strong_prefix or suffix in ("开外", "出头") or m.group(0) == s):
            continue
        lo, hi = _apply_zh_modifiers(float(val_int), prefix, suffix)
        priority = 2 if has_unit else (1 if has_strong_prefix else 0)
        if best_zh is None or priority > best_zh[0]:
            best_zh = (priority, lo, hi)
    if best_zh is not None:
        sig = _make_signal(best_zh[1], best_zh[2], is_vague=False, parse_kind="zh_num")
        if sig is not None:
            return sig

    # 11. 英文年代短语（in his thirties / forties / fifties）
    m_dec = _EN_DECADE_PHRASE_RE.search(s)
    if m_dec:
        mod = (m_dec.group(1) or "").lower()
        dec = m_dec.group(2).lower()
        lo, hi = _EN_DECADES[dec]
        if mod == "early":
            hi = min(hi, lo + 3.0)
        elif mod == "mid":
            lo, hi = lo + 3.0, min(hi, lo + 6.0)
        elif mod == "late":
            lo = max(lo, hi - 3.0)
        return _make_signal(lo, hi, is_vague=False, parse_kind="en_decade")

    # 12. 英文区间（fifteen or sixteen）
    for m in _EN_RANGE_RE.finditer(s):
        after = s[m.end():]
        if _EN_NON_AGE_FOLLOW_RE.match(after):
            continue
        if strict_unit and not _EN_AGE_CUE_RE.search(s):
            continue
        a = _en_to_int(m.group(1))
        b = _en_to_int(m.group(2))
        if a is not None and b is not None and b >= a:
            sig = _make_signal(float(a), float(b), is_vague=False, parse_kind="en_range")
            if sig is not None:
                return sig

    # 13. 英文单词／复合数字（three-and-twenty / twenty-seven / fifteen..ninety）
    for m in _EN_MOD_NUM_RE.finditer(s):
        mod_pre = re.sub(r"\s+", " ", (m.group(1) or "").strip().lower())
        word = m.group(2)
        mod_post = re.sub(r"\s+", " ", (m.group(3) or "").strip().lower())
        after = s[m.end():]
        if _EN_NON_AGE_FOLLOW_RE.match(after):
            continue
        val_int = _en_to_int(word)
        if val_int is None:
            continue
        if strict_unit and not (mod_post or mod_pre in (
            "almost", "nearly", "aged", "age", "at the age of",
            "little more than", "cannot be more than", "not more than", "about", "not", "at", "of",
        ) or _EN_AGE_CUE_RE.search(s)):
            continue
        if not strict_unit and val_int < 10 and not mod_post and not mod_pre:
            continue
        base = float(val_int)
        if mod_pre in ("almost", "nearly"):
            lo, hi = (max(0.0, base - 2.0), max(0.0, base - 0.1))
        elif mod_pre in ("under", "less than", "not"):
            lo, hi = (max(0.0, base - 2.0), max(0.0, base - 0.5))
        elif mod_pre in ("not more than", "cannot be more than"):
            lo, hi = (max(0.0, base - 2.0), base)
        elif mod_pre in ("little more than", "just over", "over"):
            lo, hi = (base + 0.1, base + 1.5)
        elif mod_post == "or more":
            lo, hi = (base, base + 5.0)
        else:
            lo, hi = (base, base)
        sig = _make_signal(lo, hi, is_vague=False, parse_kind="en_num")
        if sig is not None:
            return sig

    return None


# ---------------------------------------------------------------------------
# 定性词回退（5 级桶具体词优先于泛青年词）
# ---------------------------------------------------------------------------

_QUALITATIVE_SPECIFIC: tuple[tuple[int, float, float, re.Pattern[str]], ...] = (
    (
        0, 6.0, 11.0,
        re.compile(
            r"(?:幼童|孩童|婴儿|幼儿|幼年|童年|小孩|小童|童子|女童|男童|稚童"
            r"|\binfant\b|\btoddler\b|\blittle\s+(?:boy|girl)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        1, 14.0, 17.0,
        re.compile(
            r"(?:少年|少女|妙龄|\bteenager\b|\badolescent\b)",
            re.IGNORECASE,
        ),
    ),
    (
        2, 20.0, 26.0,
        re.compile(
            r"(?:青年男子|青年女子|年轻男子|年轻女子|青年|小伙|\byoung\s+(?:man|woman)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        3, 35.0, 49.0,
        re.compile(
            r"(?:中年|盛年|壮年|中旬|\bmiddle[- ]aged\b)",
            re.IGNORECASE,
        ),
    ),
    (
        4, 60.0, 75.0,
        re.compile(
            r"(?:年老|年迈|老人|老翁|老妪|老者|老汉|老妇|年高|年纪高大|年纪高迈|老年"
            r"|\belderly\b|\bold\s+(?:man|woman|second\s+mate|whale-hunter|cove)\b"
            r"|\b(?:very\s+old|am\s+old|look\s+(?:very\s+)?old)\b)",
            re.IGNORECASE,
        ),
    ),
)

_QUALITATIVE_VAGUE_YOUTH_RE = re.compile(
    r"(?:年纪轻轻|年纪轻|很年轻|年轻|年纪不大|年少|\bvery\s+young\b|\byouthful\b)",
    re.IGNORECASE,
)


def _parse_qualitative_text(text: str) -> AgeSignal | None:
    s = (text or "").strip()
    if not s:
        return None
    for _, lo, hi, pat in _QUALITATIVE_SPECIFIC:
        if pat.search(s):
            return _make_signal(lo, hi, is_vague=False, parse_kind="qualitative_bucket")
    if _QUALITATIVE_VAGUE_YOUTH_RE.search(s):
        return AgeSignal(
            low=13.0,
            high=29.0,
            bucket_low=1,
            bucket_high=2,
            is_vague=True,
            parse_kind="qualitative_vague_youth",
        )
    return None


def parse_age_signal(
    value: str,
    quote: str,
    *,
    context_before: str = "",
    context_after: str = "",
) -> AgeSignal | None:
    """解析单条 `person.age` 事实为凡人视觉人生阶段区间信号；触发守卫或非凡人年龄返回 `None`。"""
    val = (value or "").strip()
    quo = (quote or "").strip()
    if not val and not quo:
        return None

    # 守卫 1a：相对比较与器物存续时长（仅查 value / quote，不查周边上下文防误杀）
    if _RELATIVE_OR_DURATION_RE.search(val) or _RELATIVE_OR_DURATION_RE.search(quo):
        return None

    # 守卫 1b：倒叙回忆与追述往事
    if val and (_FLASHBACK_VALUE_END_RE.search(val) or _FLASHBACK_VALUE_RE.search(val)):
        return None
    if quo:
        if _FLASHBACK_QUOTE_EN_RE.search(quo):
            return None
        if _FLASHBACK_QUOTE_ZH_RE.search(quo) and not _PRESENT_ANCHOR_ZH_RE.search(quo):
            return None

    # 守卫 2：显式易容伪装（value / quote / 同自然段 ±80 码点上下文）
    cb_para = (context_before or "").replace("\r", "").split("\n")[-1][-80:]
    ca_para = (context_after or "").replace("\r", "").split("\n")[0][:80]
    for piece in (val, quo, cb_para, ca_para):
        if piece and _DISGUISE_RE.search(piece):
            return None

    # 数值优先解析：先 value，仅当 value 无数值信息时回退 quote
    if val:
        res = _parse_numeric_text(val, strict_unit=False)
        if res is _SUPERNATURAL:
            return None
        if isinstance(res, AgeSignal):
            return res

    if quo:
        res_q = _parse_numeric_text(quo, strict_unit=True)
        if res_q is _SUPERNATURAL:
            return None
        if isinstance(res_q, AgeSignal):
            return res_q

    # 定性词回退：先 value，再 quote（具体 5 级桶优先于泛青年词）
    if val:
        q_val = _parse_qualitative_text(val)
        if q_val is not None:
            return q_val
    if quo:
        q_quo = _parse_qualitative_text(quo)
        if q_quo is not None:
            return q_quo
    return None


def _build_body_seq(chapters: list[dict]) -> dict[str, int]:
    body_seq: dict[str, int] = {}
    seq = 0
    for ch in chapters:
        if ch.get("kind") == "chapter":
            seq += 1
            body_seq[str(ch["id"])] = seq
    return body_seq


def _is_covered_by_existing_split(
    last_prev_age: dict,
    f_new: dict,
    existing_split_events: list[dict],
    kept_facts: list[dict],
) -> bool:
    prev_start = int(last_prev_age["start"])
    new_start = int(f_new["start"])
    new_ch = f_new.get("chapter_id")
    for ev in existing_split_events:
        if ev.get("stage_split") is not True:
            continue
        ev_start = int(ev.get("cut_start", ev["start"]))
        has_before = any(int(gf["start"]) < ev_start for gf in kept_facts)
        has_after = any(int(gf["start"]) >= ev_start for gf in kept_facts)
        if not (has_before and has_after):
            continue
        if prev_start < ev_start <= new_start:
            return True
        # 同章且两者间无其他事实
        if new_ch is not None and ev.get("chapter_id") == new_ch:
            lo = min(ev_start, new_start)
            hi = max(ev_start, new_start)
            if not any(
                lo < int(gf["start"]) < hi and str(gf.get("id")) != str(f_new.get("id"))
                for gf in kept_facts
            ):
                return True
    return False


def _align_dominant_gap_cut(
    last_prev_age: dict,
    f_new: dict,
    kept_facts: list[dict],
    body_seq: dict[str, int],
) -> tuple[int, Any]:
    """主导长断档对齐：若新旧年龄间存在唯一主导 ≥10 章空窗，将切界前移至重登场首条事实。"""
    lo = int(last_prev_age["start"])
    hi = int(f_new["start"])
    between = [gf for gf in kept_facts if lo <= int(gf["start"]) <= hi]
    if len(between) >= 2:
        ords: list[int] = []
        all_known = True
        for gf in between:
            ch_id = gf.get("chapter_id")
            ord_val = body_seq.get(str(ch_id)) if ch_id is not None else None
            if ord_val is None:
                all_known = False
                break
            ords.append(ord_val)
        if all_known and (ords[-1] - ords[0]) >= 10:
            best_r = -1
            best_delta = -1
            for r in range(1, len(between)):
                delta = ords[r] - ords[r - 1]
                if delta > best_delta or (delta == best_delta and r > best_r):
                    best_delta = delta
                    best_r = r
            if best_r >= 1 and best_delta >= 10 and best_delta >= 2 * (ords[-1] - ords[best_r]):
                return int(between[best_r]["start"]), between[best_r].get("chapter_id")
    return int(f_new["start"]), f_new.get("chapter_id")


def infer_life_stage_cuts(
    entity_facts: list[dict],
    existing_split_events: list[dict],
    chapters: list[dict],
    *,
    text: str | None = None,
) -> list[dict]:
    """根据 `person` 已核验 `age` 事实序列推导单调跨时期人生阶段切点（不改 `state_events`）。"""
    kept_facts = sorted(
        [f for f in entity_facts if f.get("verdict") in (None, "keep")],
        key=lambda f: (int(f["start"]), int(f["end"]), str(f.get("id", ""))),
    )
    age_facts = [f for f in kept_facts if f.get("facet") == "age"]
    if len(age_facts) < 2:
        return []

    text_len = len(text) if isinstance(text, str) else 0
    parsed: list[tuple[dict, AgeSignal]] = []
    for fact in age_facts:
        s, e = int(fact["start"]), int(fact["end"])
        if isinstance(text, str) and text:
            cb = text[max(0, s - 80):s].split("\n")[-1]
            ca = text[e:min(text_len, e + 80)].split("\n")[0]
        else:
            cb, ca = "", ""
        sig = parse_age_signal(
            str(fact.get("value") or ""),
            str(fact.get("quote") or ""),
            context_before=cb,
            context_after=ca,
        )
        if sig is not None:
            parsed.append((fact, sig))

    if len(parsed) < 2:
        return []

    body_seq = _build_body_seq(chapters)

    # 预计算全局后缀最小 bucket_high（用于单调不回退一票否决）
    n = len(parsed)
    suffix_min_high = [999] * n
    running_min = 999
    for idx in range(n - 1, -1, -1):
        suffix_min_high[idx] = running_min
        bh = parsed[idx][1].bucket_high
        if bh < running_min:
            running_min = bh

    first_fact, first_sig = parsed[0]
    if first_sig.is_vague:
        cur_b = first_sig.bucket_high
        cur_mid = 21.0
        cur_is_vague = True
    else:
        cur_b = first_sig.bucket_high
        cur_mid = (first_sig.low + first_sig.high) / 2.0
        cur_is_vague = False
    cur_ch_id = first_fact.get("chapter_id")
    cur_ch = body_seq.get(str(cur_ch_id)) if cur_ch_id is not None else None
    cur_start = int(first_fact["start"])
    last_stage_age_fact = first_fact

    cuts: list[dict] = []
    for idx in range(1, n):
        fact, sig = parsed[idx]
        mid = (sig.low + sig.high) / 2.0
        ch_id = fact.get("chapter_id")
        ch_i = body_seq.get(str(ch_id)) if ch_id is not None else None
        f_start = int(fact["start"])

        if ch_i is not None and cur_ch is not None:
            cross_chapter = ch_i > cur_ch
        else:
            cross_chapter = (ch_id != cur_ch_id) and (f_start > cur_start)

        min_delta = 3.0 if sig.bucket_low <= 2 else 5.0
        can_promote = (
            (not sig.is_vague)
            and (sig.bucket_low > cur_b)
            and ((mid - cur_mid) >= min_delta)
            and cross_chapter
            and (suffix_min_high[idx] >= sig.bucket_low)
        )

        if can_promote:
            from_b = cur_b
            to_b = sig.bucket_low
            from_name = LIFE_STAGE_BUCKETS[from_b]
            to_name = LIFE_STAGE_BUCKETS[to_b]
            if not _is_covered_by_existing_split(
                last_stage_age_fact, fact, existing_split_events, kept_facts,
            ):
                cut_start, cut_ch_id = _align_dominant_gap_cut(
                    last_stage_age_fact, fact, kept_facts, body_seq,
                )
                cuts.append({
                    "id": digest([
                        "life-stage-cut-v1",
                        str(fact.get("entity_id", "")),
                        from_name,
                        to_name,
                        str(fact["id"]),
                    ]),
                    "event_type": "age_jump",
                    "trigger_source": TRIGGER_SOURCE_LIFE_STAGE,
                    "from_stage_bucket": from_name,
                    "to_stage_bucket": to_name,
                    "source_fact_id": str(fact["id"]),
                    "value": (
                        f"跨时期阶段变化：{LIFE_STAGE_BUCKET_LABELS[from_name]}"
                        f"→{LIFE_STAGE_BUCKET_LABELS[to_name]}（{fact['value']}）"
                    ),
                    "quote": str(fact["quote"]),
                    "start": int(fact["start"]),
                    "end": int(fact["end"]),
                    "chapter_id": fact.get("chapter_id"),
                    "cut_start": int(cut_start),
                    "cut_chapter_id": cut_ch_id,
                    "stage_split": True,
                })
            # 无论是否因既有事件去重而跳过合成，当前阶段状态均向前推进
            cur_b = sig.bucket_high
            cur_mid = mid
            cur_ch = ch_i
            cur_ch_id = ch_id
            cur_start = f_start
            last_stage_age_fact = fact
            cur_is_vague = False
            continue

        # 未定泛青年初态在遇到首个明确落在 [1, 2] 内的非模糊年龄时无损收窄而不切分
        if cur_is_vague and (not sig.is_vague) and (1 <= sig.bucket_low <= 2) and (sig.bucket_high <= 2):
            cur_b = sig.bucket_high
            cur_mid = mid
            cur_ch = ch_i
            cur_ch_id = ch_id
            cur_start = f_start
            last_stage_age_fact = fact
            cur_is_vague = False
            continue

        # 同桶或兼容年龄事实持续更新本阶段末条年龄锚点
        if sig.bucket_low <= cur_b <= sig.bucket_high:
            last_stage_age_fact = fact

    return cuts
