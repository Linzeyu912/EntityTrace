from __future__ import annotations

import bisect
import hashlib
import re
from pathlib import Path

from .storage import Store
from .util import digest, dumps

OFFSET_UNIT = "unicode_code_point"

# 句界安全切分：一级句末标点、二级句读与空白、闭合标点与英文常见缩写。
PRIMARY_END = frozenset("。！？；.!?;")
SECONDARY_END = frozenset("，,、 \t")
CLOSING_PUNCT = frozenset("”’\"')）】」』")
EN_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "vs", "etc",
    "no", "vol", "ch", "pp", "esq", "messrs", "mme", "mlle",
})

# 非正文噪声行：链接、推广、整理模板、页码导航、作者题外话与跨章重复尾注。
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\S+\.(?:com|cn|net|org|cc|me|io|xyz|top|vip|club)(?![A-Za-z0-9])",
    re.I,
)
PROMO_RE = re.compile(
    r"公众号|微信号|QQ\s?群|书友群|读者群|扫码|二维码|加群|下载\s?APP|百度搜索|"
    r"最新章节|全文阅读|继续阅读|求(?:月票|推荐票|收藏|打赏|订阅|关注)|"
    r"(?:感谢|谢谢|多谢).{0,8}(?:盟主|舵主|堂主|打赏|月票|推荐票)|"
    r"(?:盟主|舵主|堂主).{0,8}(?:打赏|万赏)"
)
TEMPLATE_PATTERNS = tuple(re.compile(p) for p in (
    r"本书由.{1,20}整理",
    r"更多.{1,30}(?:请访问|请到|请搜索|请关注)",
    r"本(?:文|书|章)来自",
    r"(?:免费|正版).{0,10}(?:小说|阅读)",
    r"(?:转载|搬运).{0,10}(?:请注明|出处)",
    r"(?:起点|纵横|17k|晋江|红袖|潇湘|飞卢|番茄).{0,20}(?:首发|原创|独家|连载)",
    r"(?:txt|TXT).{0,10}(?:下载|全集|全本)",
    r"(?:新书|新作).{0,10}(?:求收藏|求推荐|求月票|求打赏)",
    r"(?:收藏|推荐|月票|打赏).{0,5}(?:感谢|谢谢|多谢)",
))
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
NAV_CHAPTER_RE = re.compile(r"^\s*(?:上一章|下一章|返回目录|章节目录|目录)\s*$")
AUTHOR_NOTE_START = re.compile(
    r"^[\s]*[（(]?(?:作者|笔者|写手|[Pp][Ss])[：:，,、\s]?"
    r"(?:的话|题外话|感言|留言|请假|请假条|请假通知|停更|断更|今日|昨天|昨天一|今日一|大家|唠|碎碎念|叨叨)"
)
AUTHOR_NOTE_END = re.compile(r"^[\s]*[）)]?\s*(?:以上|完毕|结束|over|就这样|谢谢|感谢)[\s。.]*$", re.I)

# 脏称呼语法判定：句中标点、黏连叙述动作词、超长残片（神位官阶后缀豁免）。
SENTENCE_PUNCT_RE = re.compile(r"[，。！？；：]")
ACTION_SUFFIX_RE = re.compile(
    r"(?:说道|问道|笑道|怒道|喝道|叫道|答道|骂道|叹道|看着|望着|拿着|走出|走进|走了|来到|只见|心中|暗道|心想)$"
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
BOOK_MARK_RE = re.compile(r"[《》·]")
DEITY_SUFFIX_RE = re.compile(r"(?:天君|真君|元君|天尊)(?:之神)?$")


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


def _abbreviation_dot(line: str, index: int) -> bool:
    """判断英文句点是否落在常见缩写词内（Mr. / Mrs. / St. …），避免劈开多词专名。"""
    left = index - 1
    while left >= 0 and line[left].isascii() and line[left].isalpha():
        left -= 1
    return line[left + 1:index].lower() in EN_ABBREVIATIONS


def _capitalized_word_join(line: str, index: int) -> bool:
    """判断空白两侧是否为两个首字母大写的英文词（如 Elizabeth Bennet），是则不宜在此切断。"""
    prev = index - 1
    nxt = index + 1
    if prev < 0 or nxt >= len(line):
        return False
    if not (line[prev].isascii() and line[prev].isalnum() and line[nxt].isascii() and line[nxt].isalnum()):
        return False
    word_start = prev
    while word_start > 0 and line[word_start - 1].isascii() and line[word_start - 1].isalnum():
        word_start -= 1
    return line[word_start].isupper() and line[nxt].isupper()


def line_segments(line: str, limit: int = 500) -> list[tuple[int, int]]:
    """单物理行内的句界安全切分，返回相对半开区间并严格无缝覆盖整行。"""
    if limit <= 0:
        raise ValueError("limit 须为正整数")
    n = len(line)
    if n == 0:
        return [(0, 0)]
    segments: list[tuple[int, int]] = []
    offset = 0
    while offset < n:
        if n - offset <= limit:
            segments.append((offset, n))
            break
        low, high = offset + limit // 2, offset + limit
        cut = None
        for i in range(high - 1, low - 1, -1):
            if line[i] not in PRIMARY_END:
                continue
            if line[i] == "." and _abbreviation_dot(line, i):
                continue
            cut = i + 1
            while cut < high and line[cut] in CLOSING_PUNCT:
                cut += 1
            break
        if cut is None:
            for i in range(high - 1, low - 1, -1):
                if line[i] not in SECONDARY_END:
                    continue
                if line[i].isspace() and _capitalized_word_join(line, i):
                    continue
                cut = i + 1
                break
        if cut is None or cut <= offset:
            cut = high
        segments.append((offset, cut))
        offset = cut
    return segments


def _physical_lines(text: str) -> list[tuple[int, int, str]]:
    lines: list[tuple[int, int, str]] = []
    cursor = 0
    for raw in text.splitlines(keepends=True):
        lines.append((cursor, cursor + len(raw), raw))
        cursor += len(raw)
    return lines


def detect_noise_spans(text: str, chapters: list[dict] | None = None) -> list[tuple[int, int]]:
    """检测非正文噪声物理行，返回升序、互不重叠的半开码点区间；不改写原文。"""
    if not text:
        return []
    lines = _physical_lines(text)
    if not lines:
        return []
    starts = [item[0] for item in lines]
    protected: set[int] = set()
    for chapter in chapters or []:
        try:
            head_start = int(chapter.get("start", 0))
            body_start = int(chapter.get("body_start", head_start))
        except (TypeError, ValueError):
            continue
        if body_start <= head_start:
            continue
        first = bisect.bisect_left(starts, head_start)
        for idx in range(first, len(lines)):
            line_start, line_end, _raw = lines[idx]
            if line_start >= body_start:
                break
            if line_end > head_start:
                protected.add(idx)

    marked: dict[int, str] = {}

    def mark(index: int, category: str) -> None:
        if index in protected or index in marked:
            return
        marked[index] = category

    for index, (_start, _end, raw) in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if URL_RE.search(stripped) and len(URL_RE.sub("", stripped).strip()) <= 30:
            mark(index, "url")
            continue
        if len(stripped) <= 80 and PROMO_RE.search(stripped):
            mark(index, "promo")
            continue
        if len(stripped) <= 200 and any(pattern.search(stripped) for pattern in TEMPLATE_PATTERNS):
            mark(index, "template")
            continue
        if PAGE_NUM_RE.match(stripped) or NAV_CHAPTER_RE.match(stripped):
            mark(index, "nav")
            continue

    # 作者题外话：找到结束标记才整块屏蔽，否则只屏蔽起始行，避免吞掉正文。
    for index, (_start, _end, raw) in enumerate(lines):
        stripped = raw.strip()
        if not stripped or len(stripped) >= 80:
            continue
        if not AUTHOR_NOTE_START.match(stripped):
            continue
        close = None
        for following in range(index + 1, min(index + 11, len(lines))):
            if AUTHOR_NOTE_END.match(lines[following][2].strip()):
                close = following
                break
        if close is None:
            mark(index, "author_note")
        else:
            for cursor in range(index, close + 1):
                mark(cursor, "author_note")

    # 跨章重复尾注：章节数不少于四，同一尾行出现章数达到章节数一半即视为噪声。
    # 若某章正文只有一个非空物理行，该行是整段正文而非尾注，不参与重复尾注统计，避免吞掉正文。
    if chapters and len(chapters) >= 4:
        chapter_counts: dict[str, int] = {}
        for chapter in chapters:
            try:
                chapter_start = int(chapter.get("start", 0))
                chapter_end = int(chapter.get("end", len(text)))
                body_start = max(int(chapter.get("body_start", chapter_start)), chapter_start)
            except (TypeError, ValueError):
                continue
            begin = bisect.bisect_left(starts, body_start)
            finish = bisect.bisect_left(starts, chapter_end)
            body_lines = [(raw.strip()) for _ls, _le, raw in lines[begin:finish] if raw.strip()]
            if len(body_lines) < 2:
                continue
            seen: set[str] = set()
            for _line_start, _line_end, raw in lines[max(begin, finish - 5):finish]:
                stripped = raw.strip()
                if len(stripped) > 3:
                    seen.add(stripped)
            for stripped in seen:
                chapter_counts[stripped] = chapter_counts.get(stripped, 0) + 1
        threshold = len(chapters) * 0.5
        repeated = {value for value, count in chapter_counts.items() if count >= threshold}
        if repeated:
            for index, (_start, _end, raw) in enumerate(lines):
                if raw.strip() in repeated:
                    mark(index, "repeated_tail")

    return [(lines[index][0], lines[index][1]) for index in sorted(marked) if lines[index][2].strip()]


def dirty_surface_reasons(surface: str) -> tuple[str, ...]:
    """判定称呼是否为语法上不可能成立的脏残片；返回命中的原因元组。"""
    value = surface.strip()
    if not value:
        return ()
    reasons: list[str] = []
    if SENTENCE_PUNCT_RE.search(value):
        reasons.append("sentence_punct")
    if len(value) >= 4 and ACTION_SUFFIX_RE.search(value):
        reasons.append("action_suffix")
    if (
        len(CJK_RE.findall(value)) > 12
        and not BOOK_MARK_RE.search(value)
        and not DEITY_SUFFIX_RE.search(value)
    ):
        reasons.append("long_fragment")
    return tuple(reasons)


def is_dirty_surface(surface: str) -> bool:
    return bool(dirty_surface_reasons(surface))


def units(
    text: str,
    start: int = 0,
    end: int | None = None,
    limit: int = 700,
    skip_spans: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> list[dict]:
    """按原始物理行和句界安全长度产生定位单元，不改写原文，可整行跳过噪声区间。"""
    end = len(text) if end is None else end
    spans = tuple(skip_spans or ())
    span_starts = [span[0] for span in spans]
    output: list[dict] = []
    cursor = start
    for line in text[start:end].splitlines(keepends=True):
        line_start, line_end = cursor, cursor + len(line)
        if spans:
            index = bisect.bisect_left(span_starts, line_end) - 1
            if index >= 0 and spans[index][0] < line_end and line_start < spans[index][1]:
                cursor = line_end
                continue
        for rel_a, rel_b in line_segments(line, limit):
            a, b = line_start + rel_a, line_start + rel_b
            if text[a:b].strip():
                output.append({"id": f"u{a}_{b}", "start": a, "end": b, "text": text[a:b],
                               "complete_line": rel_a == 0 and rel_b == len(line)})
        cursor = line_end
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
