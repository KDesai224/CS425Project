"""Step 5: grow the extracted span to its whole clause (40 to about 300 words).

expand_to_clause is copied unchanged from Grpproj/scripts/02_build_pipeline_set.py.
The same function built the CUAD and ContractNLI sources of Gemma's synthetic
training pairs (Grpproj step 7), so the simplifier sees passages shaped like the
ones it was trained on. clause_span is the same logic, also returning where the
passage sits in the text, for page and section lookup; tests/test_expand.py
checks that the two always give the same passage.
"""
import re

from .config import CLAUSE_MAX_WORDS as MAX_WORDS
from .config import CLAUSE_MIN_WORDS as MIN_WORDS


def expand_to_clause(text, start, end):
    """Grow [start, end) to whole paragraphs until the passage has MIN_WORDS words,
    then cut it back to MAX_WORDS around the span if it is too long."""
    blocks = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]
    inside = [i for i, (s, e) in enumerate(blocks) if s < end and e > start]
    if not inside:
        return None
    lo, hi = inside[0], inside[-1]
    words = lambda: len(text[blocks[lo][0] : blocks[hi][1]].split())
    while words() < MIN_WORDS and (lo > 0 or hi < len(blocks) - 1):
        if hi < len(blocks) - 1:
            hi += 1
        if words() < MIN_WORDS and lo > 0:
            lo -= 1
    passage_start, passage_end = blocks[lo][0], blocks[hi][1]
    if len(text[passage_start:passage_end].split()) > MAX_WORDS:
        # Keep the span and add context on both sides up to the limit, starting
        # and ending on a sentence boundary where there is one.
        budget = max(0, MAX_WORDS - len(text[start:end].split())) // 2
        before = " ".join(text[passage_start:start].split(" ")[-budget:]) if budget else ""
        after = " ".join(text[end:passage_end].split(" ")[:budget]) if budget else ""
        before = re.sub(r"^.*?[.;:]\s+(?=[A-Z(\"])", "", before, count=1, flags=re.S)
        last_stop = after.rfind(". ")
        if not after.rstrip().endswith(".") and last_stop >= 0:
            after = after[: last_stop + 1]
        return (before + text[start:end] + after).strip()
    return re.sub(r"\s*\n\s*", "\n", text[passage_start:passage_end]).strip()


def clause_span(text, start, end):
    """(passage, passage_start, passage_end): expand_to_clause's passage and its offsets in text.

    The passage is text[passage_start:passage_end] with surrounding whitespace removed,
    and, when whole paragraphs are kept, the whitespace around line breaks collapsed.
    """
    blocks = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]
    inside = [i for i, (s, e) in enumerate(blocks) if s < end and e > start]
    if not inside:
        return None
    lo, hi = inside[0], inside[-1]
    words = lambda: len(text[blocks[lo][0] : blocks[hi][1]].split())
    while words() < MIN_WORDS and (lo > 0 or hi < len(blocks) - 1):
        if hi < len(blocks) - 1:
            hi += 1
        if words() < MIN_WORDS and lo > 0:
            lo -= 1
    passage_start, passage_end = blocks[lo][0], blocks[hi][1]
    if len(text[passage_start:passage_end].split()) > MAX_WORDS:
        budget = max(0, MAX_WORDS - len(text[start:end].split())) // 2
        # Both pieces are exact substrings: the tail of text[passage_start:start] and the
        # head of text[end:passage_end] (splitting on " " and re-joining loses nothing).
        before = " ".join(text[passage_start:start].split(" ")[-budget:]) if budget else ""
        after = " ".join(text[end:passage_end].split(" ")[:budget]) if budget else ""
        before = re.sub(r"^.*?[.;:]\s+(?=[A-Z(\"])", "", before, count=1, flags=re.S)
        last_stop = after.rfind(". ")
        if not after.rstrip().endswith(".") and last_stop >= 0:
            after = after[: last_stop + 1]
        s, e = start - len(before), end + len(after)
        passage = text[s:e]
    else:
        s, e = passage_start, passage_end
        passage = re.sub(r"\s*\n\s*", "\n", text[s:e])
    # strip() removes all leading/trailing whitespace from the passage, so the offsets
    # move to the first and last non-space characters of the raw slice.
    raw = text[s:e]
    s += len(raw) - len(raw.lstrip())
    e -= len(raw) - len(raw.rstrip())
    return passage.strip(), s, e
