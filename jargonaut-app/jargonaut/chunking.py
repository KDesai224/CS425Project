"""Step 1b: split a contract into overlapping word windows.

The same algorithm as data_prep/scripts/make_chunks.py (Theo), so BM25 and
DeBERTa see the same kind of passage they were tuned and trained on: windows of
300 whitespace-separated words, each starting 240 words after the previous one
(60 words of overlap). A chunk runs from the first character of its first word
to the last character of its last word.
"""
import re
from dataclasses import dataclass

from .config import CHUNK_MAX_WORDS, CHUNK_OVERLAP


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    char_start: int
    char_end: int      # exclusive
    text: str


def chunk_spans(text, max_units=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    """(start, end) character offsets of each window."""
    assert 0 <= overlap < max_units
    units = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    stride = max_units - overlap
    spans, i = [], 0
    while True:
        window = units[i:i + max_units]
        if not window:
            break
        spans.append((window[0][0], window[-1][1]))
        if i + max_units >= len(units):
            break
        i += stride
    return spans


def chunk_text(text, doc_id, max_units=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    return [Chunk(f"{doc_id}_c{k:04d}", k, s, e, text[s:e])
            for k, (s, e) in enumerate(chunk_spans(text, max_units, overlap))]
