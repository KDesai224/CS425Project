"""Step 1a: read an uploaded contract (PDF, DOCX or TXT) into plain text, keeping
page numbers and section headings on the side.

The text is laid out the way CUAD's contracts are, since that is what BM25 and
DeBERTa were tuned and trained on: one paragraph per line, wrapped lines joined
back together, running headers, footers and page numbers removed. Pages and
headings are stored as character offsets into that text, so any span found by
the later steps can be traced back to its page and section.

Not supported: scanned PDFs without a text layer (no OCR), legacy .doc files,
and multi-column layouts (read column by column only if the PDF orders them so).
"""
import io
import re
import statistics
import zipfile
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


class IngestError(ValueError):
    """The file cannot be turned into text; the message is shown to the user."""


@dataclass
class Heading:
    start: int     # offset of the heading's paragraph in Document.text
    level: int     # 1 = outermost
    text: str


@dataclass
class Document:
    name: str
    kind: str                     # "pdf", "docx" or "txt"
    text: str
    page_starts: list | None      # page_starts[i] = offset where page i + 1 starts; None if unknown
    headings: list
    warnings: list = field(default_factory=list)

    @property
    def n_pages(self):
        return len(self.page_starts) if self.page_starts else None

    def page_at(self, offset):
        return bisect_right(self.page_starts, offset) if self.page_starts else None

    def pages(self, start, end):
        """Page numbers (1-based) covered by text[start:end], or None if pages are unknown."""
        if not self.page_starts:
            return None
        return list(range(self.page_at(start), self.page_at(max(start, end - 1)) + 1))

    def section(self, offset):
        """The headings in force at offset, outermost first."""
        stack = []
        for h in self.headings:
            if h.start > offset:
                break
            while stack and stack[-1].level >= h.level:
                stack.pop()
            stack.append(h)
        return stack


# --- Headings ------------------------------------------------------------------

_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to",
                "with", "upon", "under", "its", "their", "this", "that", "not", "no"}
_DIVISION = re.compile(r"^(ARTICLE|Article|SECTION|Section|PART|Part|SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex"
                       r"|APPENDIX|Appendix)\s+(\d+(?:\.\d+)*|[IVXLC]+|[A-Z])\b")
_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,3})*)\.?\s+(.*)$")
_RUN_IN = re.compile(r"^([A-Z][^.;:]{1,80}?)[.:]\s+\S")


def _title_case(words):
    content = [w for w in words if w.lower().strip("()\"'“”,.:;") not in _SMALL_WORDS and any(ch.isalpha() for ch in w)]
    return bool(content) and sum(w.lstrip("(\"'“")[:1].isupper() for w in content) / len(content) >= 0.75


def _all_caps(text):
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 4 and sum(ch.isupper() for ch in letters) / len(letters) >= 0.85


def detect_heading(text, bold=False, big=False):
    """(level, heading) if this paragraph is, or starts with, a section heading; else None.

    Recognised: "ARTICLE 9 ...", "Section 9.2 ...", numbered title lines ("9. Term and
    Termination", "9.2 TERMINATION"), numbered run-in titles ("9.2 Termination for
    Convenience. Either party may ..."), and short all-caps lines. A short line in
    bold or a larger font counts too when it is in title case.
    """
    t = " ".join(text.split())
    words = t.split()
    if not words or t.endswith((",", ";")):
        return None
    m = _DIVISION.match(t)
    if m:
        number = m.group(2)
        level = number.count(".") + 1 if m.group(1).lower() == "section" and number[0].isdigit() else 1
        # "Exhibit A attached hereto ..." is a sentence, not a heading: the title must look like one
        title = t[m.end():].strip(" .:-–—")
        if len(words) <= 15 and (not title or _all_caps(title) or _title_case(title.split()) or bold):
            return level, t.rstrip(".:")
        # a run-in title: "SECTION 12. Termination. Capital Resources may terminate ..."
        rest = t[m.end():].lstrip(" .:-–—")
        r = _RUN_IN.match(rest)
        if r and len(r.group(1).split()) <= 8 and (_title_case(r.group(1).split()) or _all_caps(r.group(1))):
            return level, t[:len(t) - len(rest) + r.end(1)]
        return None
    m = _NUMBERED.match(t)
    if m:
        level, rest = m.group(1).count(".") + 1, m.group(2)
        if 0 < len(rest.split()) <= 10 and (_all_caps(rest) or _title_case(rest.split()) or bold):
            return level, t.rstrip(".:")
        r = _RUN_IN.match(rest)
        if r and len(r.group(1).split()) <= 8 and (_title_case(r.group(1).split()) or _all_caps(r.group(1))):
            return level, t[:m.start(2) + r.end(1)]
        return None
    if len(words) <= 10 and (_all_caps(t) or ((bold or big) and _title_case(words))):
        return 1, t.rstrip(".:")
    return None


_CONTINUES = re.compile(r"(?i)(?:\b(?:and|or|of|the|to|for|in|on|with|by|a|an)|[,&-])$")


def _is_heading_line(text, bold=False):
    """The whole line is a heading, not a run-in title followed by clause text, and it does
    not carry on to the next line ("... Representations and")."""
    t = " ".join(text.split())
    if _CONTINUES.search(t):
        return False
    h = detect_heading(t, bold=bold)
    return h is not None and len(h[1]) >= len(t.rstrip(".: ")) - 1


# --- Building the document -------------------------------------------------------

@dataclass
class _Para:
    text: str
    page_breaks: list = field(default_factory=list)   # offsets inside text where a new page starts
    heading: tuple | None = None                       # (level, text)


def _assemble(name, kind, paras, n_pages_known, first_page_break=True, warnings=()):
    """Join paragraphs with newlines; turn their page breaks and headings into offsets."""
    parts, page_starts, headings, length = [], [], [], 0
    if n_pages_known and first_page_break:
        page_starts.append(0)
    for p in paras:
        if not p.text.strip():
            continue
        if parts:
            parts.append("\n")
            length += 1
        page_starts.extend(length + b for b in p.page_breaks)
        if p.heading:
            headings.append(Heading(length, *p.heading))
        parts.append(p.text)
        length += len(p.text)
    text = "".join(parts)
    if not text.strip():
        raise IngestError(f"No text could be read from {name}.")
    return Document(name, kind, text, page_starts if n_pages_known else None,
                    _merge_division_titles(headings, text), list(warnings))


def _merge_division_titles(headings, text):
    """'ARTICLE 9' followed by a title line 'TERM AND TERMINATION' becomes one heading."""
    out = []
    for h in headings:
        prev = out[-1] if out else None
        if (prev and h.level == prev.level == 1 and re.fullmatch(r"(?i)(article|part|section)\s+\S+", prev.text)
                and text[prev.start:h.start].count("\n") == 1 and not _DIVISION.match(h.text)):
            out[-1] = Heading(prev.start, 1, f"{prev.text} {h.text}")
        else:
            out.append(h)
    return out


def _keep_or_drop_hyphen(left, right, vocabulary):
    """Rejoin a word hyphenated across a line break, as the document spells it elsewhere."""
    head = re.search(r"([A-Za-z]+)-$", left)
    tail = re.match(r"[A-Za-z]+", right)
    if head and tail:
        joined = (head.group(1) + tail.group(0)).lower()
        hyphenated = f"{head.group(1)}-{tail.group(0)}".lower()
        if joined in vocabulary and hyphenated not in vocabulary:
            return left[:-1] + right
    return left + right


# --- PDF -------------------------------------------------------------------------

@dataclass
class _Line:
    text: str
    bold: bool
    size: float
    x0: float
    y0: float
    y1: float
    bold_prefix: str     # leading bold text, for run-in headings


_PAGE_NUMBER = re.compile(r"(?i)^(page\s*)?[-–—]?\s*(\d{1,4}|[ivxlc]{1,6})\s*[-–—]?(\s*(of|/)\s*\d{1,4})?$")
_ITEM_START = re.compile(r"^(\(?[a-zA-Z0-9]{1,4}[.)]|\d{1,2}(\.\d{1,3})+\.?|[•▪●-])\s")
_SENTENCE_END = (".", ":", ";", "?", "!", '"', "”")
# Characters some PDF fonts produce in place of plain ones: a Greek question mark for ";"
# (common in EDGAR filings), soft hyphens, zero-width spaces and ligatures.
_PDF_CHARS = str.maketrans({";": ";", "­": None, "​": None, "﻿": None, "ﬀ": "ff",
                            "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})


def _clean(text):
    text = text.rstrip()
    if text.endswith("­"):        # a soft hyphen ending the line: rejoined like a hyphen
        text = text[:-1] + "-"
    return " ".join(text.translate(_PDF_CHARS).split())


def _same_row(a, b):
    """The two lines overlap vertically by more than half the shorter one's height."""
    return min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5 * min(a.y1 - a.y0, b.y1 - b.y0)


def _join_row(pieces):
    """Lines on the same row of a block as one line, left to right. PDFs often set a clause
    number apart from its title or text ("11."  "INTELLECTUAL PROPERTY", "(a)"  "The ...")."""
    pieces = sorted(pieces, key=lambda ln: ln.x0)
    prefix = []
    for ln in pieces:
        prefix.append(ln.bold_prefix)
        if not ln.bold:
            break
    return _Line(" ".join(ln.text for ln in pieces), all(ln.bold for ln in pieces), max(ln.size for ln in pieces),
                 pieces[0].x0, min(ln.y0 for ln in pieces), max(ln.y1 for ln in pieces),
                 " ".join(p for p in prefix if p))


def _pdf_lines(page):
    """Text lines of one page as blocks of _Line, in reading order."""
    blocks = []
    for b in page.get_text("dict", sort=True)["blocks"]:
        if b.get("type") != 0:
            continue
        rows = []
        for ln in b["lines"]:
            if abs(ln.get("dir", (1, 0))[1]) > 0.1:      # rotated margin text (e.g. filing stamps)
                continue
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            is_bold = [bool(s["flags"] & 16) or "bold" in s["font"].lower() for s in spans]
            prefix = []
            for s, bold in zip(spans, is_bold):
                if not bold:
                    break
                prefix.append(s["text"])
            text = _clean("".join(s["text"] for s in ln["spans"]))
            if not text:
                continue
            line = _Line(text, all(is_bold), max(s["size"] for s in spans), ln["bbox"][0],
                         ln["bbox"][1], ln["bbox"][3], _clean("".join(prefix)))
            row = next((r for r in rows if _same_row(r[0], line)), None)
            if row is None:
                rows.append([line])
            else:
                row.append(line)
        if rows:
            blocks.append([r[0] if len(r) == 1 else _join_row(r) for r in rows])
    return blocks


def _roman_value(s):
    values = [{"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}[ch] for ch in s.lower()]
    return sum(-v if v < w else v for v, w in zip(values, values[1:] + [0]))


def _page_numbers(pages, furniture):
    """ids of the page-number lines: a lone "7", "-7-", "Page 7", "vii" or "7 of 30" on a row of
    its own that is the first or last line of its page (not counting `furniture`), or that
    continues a run (6 shortly before it or 8 shortly after it). Web pages printed to PDF, as
    most EDGAR filings are, put several original pages on one PDF page, so their page
    numbers sit between paragraphs."""
    candidates = []      # (line, value, the page's lines) in reading order
    for _, blocks in pages:
        flat = [ln for b in blocks for ln in b]
        for ln in flat:
            m = _PAGE_NUMBER.match(ln.text)
            if m and _alone(ln, flat):
                n = m.group(2)
                candidates.append((ln, int(n) if n.isdigit() else _roman_value(n), flat))
    found = set()
    for k, (ln, value, flat) in enumerate(candidates):
        others = [o for o in flat if o is not ln and id(o) not in furniture]
        edge = all(o.y0 > ln.y0 for o in others) or all(o.y0 < ln.y0 for o in others)
        run = (any(v == value - 1 for _, v, _ in candidates[max(0, k - 3):k])
               or any(v == value + 1 for _, v, _ in candidates[k + 1:k + 4]))
        if edge or run:
            found.add(id(ln))
    return found


def _alone(ln, flat):
    """No other line of the page shares ln's row."""
    return not any(o is not ln and _same_row(o, ln) for o in flat)


def _furniture_key(text):
    return re.sub(r"\d+", "#", text.lower())


def _page_furniture(pages, page_numbers):
    """ids of lines that keep recurring next to page numbers, such as the "Source: ACME CORP,
    10-K, 3/1/2010" line EDGAR prints under each page number, and of every other copy of
    them that sits on a row of its own."""
    flats = [[ln for b in blocks for ln in b] for _, blocks in pages]
    near = Counter()
    for flat in flats:
        for i, ln in enumerate(flat):
            if (id(ln) not in page_numbers and any(id(o) in page_numbers for o in flat[max(0, i - 1):i + 2])
                    and _alone(ln, flat)):
                near[_furniture_key(ln.text)] += 1
    keys = {k for k, c in near.items() if c >= max(2, 0.3 * len(page_numbers))}
    return {id(ln) for flat in flats for ln in flat if _furniture_key(ln.text) in keys and _alone(ln, flat)}


def _read_pdf(data, name):
    try:
        import pymupdf
    except ImportError:  # PyMuPDF before 1.24
        import fitz as pymupdf
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise IngestError(f"{name} could not be opened as a PDF ({e}).") from e
    if pdf.needs_pass:
        raise IngestError(f"{name} is password-protected.")
    pages = [(page.rect.height, _pdf_lines(page)) for page in pdf]
    if sum(len(ln.text) for _, blocks in pages for b in blocks for ln in b) < 20 * len(pages):
        raise IngestError(f"{name} has no text layer (it looks scanned). OCR is not supported; "
                          "please upload a PDF with selectable text or a Word file.")

    # Running headers and footers: lines near the top or bottom edge that repeat on
    # many pages once digits are ignored, plus bare page numbers there. Page numbers
    # and the lines that recur next to them are also found in mid-page (breaks).
    def margin(ln, height):
        return ln.y1 < 0.08 * height or ln.y0 > 0.92 * height

    seen = Counter()
    for height, blocks in pages:
        seen.update({_furniture_key(ln.text) for b in blocks for ln in b if margin(ln, height)})
    repeated = {k for k, c in seen.items() if len(pages) >= 2 and c >= max(2, 0.4 * len(pages))}
    drop = {id(ln) for height, blocks in pages for b in blocks for ln in b
            if margin(ln, height) and (_furniture_key(ln.text) in repeated or _PAGE_NUMBER.match(ln.text))}
    page_numbers = _page_numbers(pages, drop)
    breaks = page_numbers | _page_furniture(pages, page_numbers)
    drop |= breaks

    sizes = [ln.size for _, blocks in pages for b in blocks for ln in b for _ in range(len(ln.text))]
    body_size = statistics.median(sizes) if sizes else 0
    all_text = " ".join(ln.text for _, blocks in pages for b in blocks for ln in b)
    vocabulary = set(re.findall(r"[a-z]+(?:-[a-z]+)*", all_text.lower()))

    paras, removed = [], 0
    for height, blocks in pages:
        page_paras, after_page_number = [], False

        def flush(current, continues):
            text = current[0].text
            for ln in current[1:]:
                text = _keep_or_drop_hyphen(text, ln.text, vocabulary) if text.endswith("-") else f"{text} {ln.text}"
            first = current[0]
            heading = detect_heading(text, bold=all(l.bold for l in current),
                                     big=body_size and first.size >= 1.15 * body_size and len(current) <= 2)
            if heading is None and first.bold_prefix and not first.bold:
                bold = first.bold_prefix.rstrip(" .:")
                if 0 < len(bold.split()) <= 10 and text.startswith(first.bold_prefix.strip()):
                    heading = detect_heading(bold, bold=True) or (2, bold)
            prev = page_paras[-1] if page_paras else None
            # A sentence cut by a page number in mid-page carries on in lower case.
            if (continues and prev and not prev.heading and not heading and text[:1].islower()
                    and not prev.text.endswith(_SENTENCE_END)):
                prev.text = f"{prev.text} {text}"
            else:
                page_paras.append(_Para(text, heading=heading))

        for block in blocks:
            current, continues = [], False
            for ln in block:
                if id(ln) in drop:
                    removed += 1
                    after_page_number = after_page_number or id(ln) in breaks
                    continue
                if current:
                    prev = current[-1]
                    after_sentence = prev.text.endswith((".", ";", ":"))
                    # a complete heading of up to three lines ends here
                    prev_is_heading = len(current) <= 3 and _is_heading_line(
                        " ".join(l.text for l in current), all(l.bold for l in current))
                    if prev_is_heading or (
                            _is_heading_line(ln.text, ln.bold) and (after_sentence or ln.bold != prev.bold)) or (
                            after_sentence and (_ITEM_START.match(ln.text) or ln.x0 > prev.x0 + 10)):
                        flush(current, continues)
                        current = []
                if not current:
                    continues = after_page_number
                after_page_number = False
                current.append(ln)
            if current:
                flush(current, continues)
        if not page_paras:     # blank page: its start is carried to the next paragraph below
            paras.append(_Para("", page_breaks=[0]))
            continue
        # If the previous page stops mid-sentence and this one carries on in lower case,
        # it is one paragraph split by the page break.
        prev = paras[-1] if paras and paras[-1].text else None
        if (prev and not prev.heading and not page_paras[0].heading and page_paras[0].text[:1].islower()
                and not prev.text.endswith(_SENTENCE_END)):
            prev.page_breaks.append(len(prev.text) + 1)
            prev.text = f"{prev.text} {page_paras[0].text}"
            page_paras = page_paras[1:]
        else:
            page_paras[0].page_breaks.insert(0, 0)
        paras.extend(page_paras)
    merged, carry = [], []
    for p in paras:
        if not p.text:
            carry.extend(p.page_breaks)
            continue
        p.page_breaks = carry + p.page_breaks
        carry = []
        merged.append(p)
    warnings = [f"Removed {removed} page numbers and running header/footer lines."] if removed else []
    doc = _assemble(name, "pdf", merged, n_pages_known=True, first_page_break=False, warnings=warnings)
    doc.page_starts += [len(doc.text)] * (len(pages) - len(doc.page_starts))   # trailing blank pages
    return doc


# --- DOCX ------------------------------------------------------------------------

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def _roman(n):
    out = ""
    for value, sym in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
                       (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= value:
            out, n = out + sym, n - value
    return out


def _letters(n):
    # Word repeats the letter: a..z, aa..zz, aaa..
    return chr(ord("a") + (n - 1) % 26) * ((n - 1) // 26 + 1)


def _format_number(n, fmt):
    if fmt in ("lowerLetter", "upperLetter"):
        s = _letters(n)
        return s.upper() if fmt == "upperLetter" else s
    if fmt in ("lowerRoman", "upperRoman"):
        s = _roman(n)
        return s.upper() if fmt == "upperRoman" else s
    if fmt == "decimalZero":
        return f"{n:02d}"
    return str(n)


class _Numbering:
    """Renders Word's automatic list numbers ("9.2", "(a)"), which are not stored in the text.

    Covers the common cases: decimal, letter and roman formats, start values, lvlText
    patterns such as "%1.%2", start overrides, and numbering inherited from styles.
    """

    def __init__(self, docx):
        self.abstract, self.num_to_abstract, self.overrides = {}, {}, {}
        self.counters, self.started = {}, set()
        try:
            root = docx.part.numbering_part.element
        except (KeyError, NotImplementedError, AttributeError):
            root = None
        if root is not None:
            for a in root.iter(W + "abstractNum"):
                levels = {}
                for lvl in a.iter(W + "lvl"):
                    get = lambda tag, default=None: (lvl.find(W + tag).get(W + "val") if lvl.find(W + tag) is not None else default)
                    levels[int(lvl.get(W + "ilvl"))] = {
                        "start": int(get("start", "1")), "fmt": get("numFmt", "decimal"),
                        "text": get("lvlText", ""), "legal": lvl.find(W + "isLgl") is not None}
                self.abstract[a.get(W + "abstractNumId")] = levels
            for num in root.iter(W + "num"):
                num_id = num.get(W + "numId")
                ref = num.find(W + "abstractNumId")
                if ref is not None:
                    self.num_to_abstract[num_id] = ref.get(W + "val")
                for o in num.iter(W + "lvlOverride"):
                    start = o.find(W + "startOverride")
                    if start is not None:
                        self.overrides[(num_id, int(o.get(W + "ilvl")))] = int(start.get(W + "val"))
        self.style_numpr, self.style_names, self.style_based_on, self.style_outline = {}, {}, {}, {}
        styles = docx.styles.element
        for s in styles.iter(W + "style"):
            sid = s.get(W + "styleId")
            name = s.find(W + "name")
            self.style_names[sid] = name.get(W + "val") if name is not None else sid
            based = s.find(W + "basedOn")
            if based is not None:
                self.style_based_on[sid] = based.get(W + "val")
            ppr = s.find(W + "pPr")
            if ppr is not None:
                numpr = ppr.find(W + "numPr")
                if numpr is not None:
                    self.style_numpr[sid] = numpr
                outline = ppr.find(W + "outlineLvl")
                if outline is not None:
                    self.style_outline[sid] = int(outline.get(W + "val"))

    def style_chain(self, sid):
        chain = []
        while sid and sid not in chain and len(chain) < 10:
            chain.append(sid)
            sid = self.style_based_on.get(sid)
        return chain

    def label(self, p, style_id):
        """The rendered number of paragraph p ("" if it is not numbered)."""
        ppr = p.find(W + "pPr")
        numpr = ppr.find(W + "numPr") if ppr is not None else None
        if numpr is None:
            numpr = next((self.style_numpr[s] for s in self.style_chain(style_id) if s in self.style_numpr), None)
        if numpr is None:
            return ""
        num_el, ilvl_el = numpr.find(W + "numId"), numpr.find(W + "ilvl")
        num_id = num_el.get(W + "val") if num_el is not None else None
        ilvl = int(ilvl_el.get(W + "val")) if ilvl_el is not None else 0
        abstract_id = self.num_to_abstract.get(num_id)
        if not num_id or num_id == "0" or abstract_id not in self.abstract:
            return ""
        levels = self.abstract[abstract_id]
        if ilvl not in levels:
            return ""
        counts = self.counters.setdefault(abstract_id, {})
        if (num_id, ilvl) in self.overrides and (num_id, ilvl) not in self.started:
            self.started.add((num_id, ilvl))
            counts[ilvl] = self.overrides[(num_id, ilvl)] - 1
        counts[ilvl] = counts.get(ilvl, levels[ilvl]["start"] - 1) + 1
        for deeper in [k for k in counts if k > ilvl]:
            del counts[deeper]
        lvl = levels[ilvl]
        if lvl["fmt"] in ("bullet", "none"):
            return ""

        def render(match):
            k = int(match.group(1)) - 1
            if k not in levels:
                return ""
            n = counts.get(k, levels[k]["start"])
            return _format_number(n, "decimal" if lvl["legal"] else levels[k]["fmt"])

        return re.sub(r"%(\d)", render, lvl["text"]).strip()


def _paragraph_runs(p):
    """Text of a w:p with page-break markers, skipping text boxes, deleted text and field codes.

    Returns (text, rendered_breaks, hard_breaks, bold_prefix, all_bold).
    """
    text, rendered, hard, bold_prefix, bolds, prefix_open = [], [], [], [], [], True
    for r in p.iter(W + "r"):
        anc, skip = r.getparent(), False
        while anc is not None and anc is not p:
            if anc.tag in (W + "txbxContent", MC_FALLBACK, W + "del"):
                skip = True
                break
            anc = anc.getparent()
        if skip:
            continue
        rpr = r.find(W + "rPr")
        b = rpr.find(W + "b") if rpr is not None else None
        bold = b is not None and b.get(W + "val") not in ("0", "false", "off")
        run_text = []
        for el in r:
            pos = sum(map(len, text)) + sum(map(len, run_text))
            if el.tag == W + "t":
                run_text.append(el.text or "")
            elif el.tag == W + "tab":
                run_text.append("\t")
            elif el.tag == W + "noBreakHyphen":
                run_text.append("-")
            elif el.tag == W + "lastRenderedPageBreak":
                rendered.append(pos)
            elif el.tag in (W + "br", W + "cr"):
                if el.get(W + "type") == "page":
                    hard.append(pos)
                else:
                    run_text.append(" ")
        piece = "".join(run_text)
        if piece.strip():
            bolds.append(bold)
            if prefix_open and bold:
                bold_prefix.append(piece)
            elif prefix_open:
                prefix_open = False
        elif prefix_open and bold_prefix:
            bold_prefix.append(piece)
        text.append(piece)
    return "".join(text), rendered, hard, "".join(bold_prefix), bool(bolds) and all(bolds)


def _iter_block_items(parent):
    """w:p and w:tbl elements of the body in order, looking inside content controls."""
    for child in parent.iterchildren():
        if child.tag in (W + "p", W + "tbl"):
            yield child
        elif child.tag in (W + "sdt", W + "customXml"):
            content = child.find(W + "sdtContent") if child.tag == W + "sdt" else child
            if content is not None:
                yield from _iter_block_items(content)


def _read_docx(data, name):
    try:
        import docx
        d = docx.Document(io.BytesIO(data))
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise IngestError(f"{name} could not be opened as a Word document ({e}).") from e
    numbering = _Numbering(d)
    raw = []    # (text, rendered breaks, hard breaks, heading)
    # Headings found by their look (bold, numbered) nest under the last heading-styled one.
    styled_level = [0]

    def add_paragraph(p):
        text, rendered, hard, bold_prefix, all_bold = _paragraph_runs(p)
        ppr = p.find(W + "pPr")
        style_el = ppr.find(W + "pStyle") if ppr is not None else None
        style_id = style_el.get(W + "val") if style_el is not None else None
        label = numbering.label(p, style_id)
        shift = len(label) + 1 if label and text.strip() else 0
        full = f"{label} {text.strip()}" if shift else text.strip()
        lead = len(text) - len(text.lstrip())
        fix = lambda offs: [max(0, min(len(full), o - lead + shift)) for o in offs]
        heading = None
        if full:
            names = [numbering.style_names.get(s, "") for s in numbering.style_chain(style_id)]
            styled = next((int(m.group(1)) for n in names if (m := re.match(r"(?i)heading\s*(\d)", n))), None)
            if styled is None and any(n.lower() == "title" for n in names):
                styled = 0      # the document title sits above every Heading 1
            outline = ppr.find(W + "outlineLvl") if ppr is not None else None
            if styled is None and outline is not None and int(outline.get(W + "val")) < 9:
                styled = int(outline.get(W + "val")) + 1
            if styled is None:
                styled = next((numbering.style_outline[s] + 1 for s in numbering.style_chain(style_id)
                               if numbering.style_outline.get(s, 9) < 9), None)
            if styled is not None and len(full.split()) <= 25:
                heading = (styled, full.rstrip(".:"))
                styled_level[0] = styled
            else:
                heading = detect_heading(full, bold=all_bold)
                prefix = bold_prefix.strip().rstrip(" .:")
                if heading is None and prefix and not all_bold and 0 < len(prefix.split()) <= 10:
                    heading = detect_heading(f"{label} {prefix}".strip(), bold=True) or (1, f"{label} {prefix}".strip())
                if heading:
                    heading = (styled_level[0] + heading[0], heading[1])
        raw.append((full, fix(rendered), fix(hard), heading))

    for item in _iter_block_items(d.element.body):
        if item.tag == W + "p":
            add_paragraph(item)
        else:   # one line per table row, cells separated by " | "
            for row in item.findall(W + "tr"):
                cells = []
                breaks_r, breaks_h = [], []
                for cell in row.findall(W + "tc"):
                    for p in cell.iter(W + "p"):
                        t, rendered, hard, _, _ = _paragraph_runs(p)
                        if t.strip():
                            base = sum(len(c) + 3 for c in cells)
                            breaks_r += [base + o for o in rendered]
                            breaks_h += [base + o for o in hard]
                            cells.append(t.strip())
                if cells:
                    raw.append((" | ".join(cells), breaks_r, breaks_h, None))

    has_rendered = any(r for _, r, _, _ in raw)
    paras = [_Para(t, page_breaks=sorted(set(r)) if has_rendered else [], heading=h) for t, r, _, h in raw]
    warnings = [] if has_rendered else [
        "This Word file has no saved page layout, so page numbers are not available. Sections are still shown."]
    return _assemble(name, "docx", paras, n_pages_known=has_rendered, warnings=warnings)


# --- TXT -------------------------------------------------------------------------

def _read_txt(data, name):
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pages = text.split("\f")
    paras = []
    for i, page in enumerate(pages):
        page_paras = [_Para(line.strip(), heading=detect_heading(line)) for line in page.split("\n") if line.strip()]
        if page_paras and len(pages) > 1:
            page_paras[0].page_breaks.append(0)
        paras.extend(page_paras)
    return _assemble(name, "txt", paras, n_pages_known=len(pages) > 1, first_page_break=False)


# --- Entry point -------------------------------------------------------------------

MAX_BYTES = 50 * 1024 * 1024


def read_document(data, name):
    """Read a contract from its bytes. The extension of `name` picks the reader."""
    if len(data) > MAX_BYTES:
        raise IngestError(f"{name} is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        return _read_pdf(data, name)
    if suffix == ".docx":
        return _read_docx(data, name)
    if suffix in (".txt", ".text", ".md"):
        return _read_txt(data, name)
    if suffix == ".doc":
        raise IngestError("Old .doc files are not supported. Save the file as .docx or PDF and upload it again.")
    raise IngestError(f"Unsupported file type '{suffix}'. Upload a PDF, DOCX or TXT file.")


def read_path(path):
    path = Path(path)
    return read_document(path.read_bytes(), path.name)
