"""Step 7: check that the rewrite kept the clause's meaning before it is shown.

The number, negation and NLI functions are copied unchanged from
Grpproj/jargonaut/metrics.py, which scored every Gemma run, so a rewrite passes or
fails here for the same reasons it scored well or badly in evaluation. Three checks:

  numbers   every quantity in the clause (amounts, periods, percentages, years) is in
            the rewrite, and the rewrite adds none of its own
  negation  if the clause negates something ("not", "no", "unless", "except", ...),
            the rewrite still has a negation
  nli       the rewrite entails the clause's sentences on average (NLI coverage) at
            least NLI_COVERAGE_MIN; low coverage means a condition or exception was dropped

If any check fails the user sees the original clause with a warning instead.
"""
import re

import textstat

from .config import NLI_COVERAGE_MIN, NLI_ID

# --- Copied from Grpproj/jargonaut/metrics.py ---------------------------------------

_NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fourteen": "14", "fifteen": "15",
    "twenty": "20", "thirty": "30", "forty-five": "45", "sixty": "60", "ninety": "90",
}
_LEADING_NUMBERING = re.compile(r"^\s*\(?\d+(\.\d+)*\)?\.?\s*(?=[A-Za-z(])")
# Clause headings inside a passage: "14.2 STAAR shall", "7. Amendments.", '1.21 "Products" means'.
# A dotted number, or a one- or two-digit integer followed by a period, preceded by whitespace
# and followed by a capital letter, quote or bracket. Years, "$1.5 Million" and "(30) Days" are
# not matched: the integer form needs a trailing period and nothing may precede the number but space.
_CLAUSE_HEADING = re.compile(r"(?<![\w$.,/])(?:\d+(?:\.\d+)+\.?|(?!(?:19|20)\d\d\.)\d{1,2}\.)\s+(?=[A-Z\"“(])")
# Cross-references, including lists: "Sections 6.2, 6.3, 6.4, and 6.5", "Articles 7, 8 and 9",
# "Section 1.21 through 1.24", "(Chapter 10)". A continuation number followed by a unit word,
# as in "Section 7 and 9 days", is a quantity and is kept.
_REF_WORD = (r"\b(?:sections?|articles?|clauses?|paragraphs?|sub-?sections?|sub-?paragraphs?|exhibits?|schedules?"
             r"|annex(?:es)?|appendi(?:x|ces)|chapters?|items?|rules?|parts?)")
_NOT_UNIT = (r"(?!\s*(?:\(\d+\)\s*)?(?:(?:calendar|business|working|consecutive)\s+)?"
             r"(?:days?|months?|years?|weeks?|hours?|minutes?|percent|dollars?|times|copies|units?)\b)(?!\s*%)")
_CROSS_REFERENCE = re.compile(
    _REF_WORD + r"\s+\d+(?:\.\d+)*"
    + r"(?:(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|through|to|&)\s+|\s*[-–]\s*)(?:" + _REF_WORD + r"\s+)?\d+(?:\.\d+)*"
    + _NOT_UNIT + r")*",
    re.I,
)
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_NEGATION = re.compile(r"\b(not|no|never|neither|nor|without|unless|except|cannot)\b|n't", re.I)
# Phrases that contain a negation word but negate nothing. "not less than thirty (30) days"
# is faithfully rewritten as "at least 30 days", so these do not make a source count as negated.
_NEGATION_IDIOMS = re.compile(
    r"\b(?:without\s+limit(?:ation|ing)|but\s+not\s+limited\s+to|not?\s+(?:less|later|more|earlier|fewer)\s+than"
    r"|not\s+to\s+exceed)\b", re.I
)


def numbers(text):
    """Quantities in the text (amounts, periods, percentages, years).

    Clause numbering, inline clause headings ("14.2 Licensee shall") and
    cross-references ("Section 8.1", "Sections 6.2, 6.3 and 6.4") are ignored,
    because a rewrite can drop them without changing the meaning. Number words
    are mapped to digits so "thirty (30) days" and "30 days" compare equal.
    """
    text = _CLAUSE_HEADING.sub(" ", _CROSS_REFERENCE.sub(" ", _LEADING_NUMBERING.sub("", text)))
    found = {m.replace(",", "") for m in _NUMBER.findall(text)}
    lowered = text.lower()
    found |= {digit for word, digit in _NUMBER_WORDS.items() if re.search(rf"\b{word}\b", lowered)}
    return found


def number_recall(source, output):
    """Share of the source's quantities that survive in the output. None if the source has none."""
    src = numbers(source)
    return len(src & numbers(output)) / len(src) if src else None


def number_precision(source, output):
    """Share of the output's quantities that come from the source. Low means invented numbers."""
    out = numbers(output)
    return len(out & numbers(source)) / len(out) if out else None


def negation_kept(source, output):
    """1 if a source containing a negation cue yields an output that also has one. None if no cue."""
    if not _NEGATION.search(_NEGATION_IDIOMS.sub(" ", source)):
        return None
    return 1.0 if _NEGATION.search(output) else 0.0


def split_sentences(text):
    """Rough sentence splitter; fragments under four words are merged into the previous sentence."""
    sentences = []
    for line in text.split("\n"):
        for piece in re.split(r"(?<=[.!?;])\s+(?=[A-Z(\"“])", line.strip()):
            piece = piece.strip(" -•*")
            if not piece:
                continue
            if sentences and len(piece.split()) < 4:
                sentences[-1] += " " + piece
            else:
                sentences.append(piece)
    return sentences


class NLIScorer:
    """Two-way entailment between a source clause and its rewrite.

    nli_supported   : mean P(source entails output sentence)   - low means added content
    nli_contradicted: share of output sentences the source contradicts
    nli_coverage    : mean P(output entails source sentence)   - low means dropped content
    """

    def __init__(self, model_name=NLI_ID, batch_size=8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail, self.contra = labels["entailment"], labels["contradiction"]
        self.batch_size = batch_size

    def _encode(self, batch, truncation):
        return self.tokenizer(
            [p for p, _ in batch], [h for _, h in batch],
            return_tensors="pt", padding=True, truncation=truncation, max_length=1024,
        ).to(self.device)

    def _probs(self, pairs):
        probs = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            try:
                enc = self._encode(batch, "only_first")
            except Exception:
                # A single "sentence" longer than the window (a run-on legal sentence) cannot
                # be fitted by trimming the premise alone; trim both sides instead, as the
                # gemma-eval notebook does. Only these batches are affected.
                enc = self._encode(batch, "longest_first")
            with self.torch.inference_mode():
                probs.extend(self.model(**enc).logits.softmax(-1).tolist())
        return probs

    def score(self, source, output):
        out_sents, src_sents = split_sentences(output), split_sentences(source)
        if not out_sents or not src_sents:
            return {"nli_supported": None, "nli_contradicted": None, "nli_coverage": None}
        forward = self._probs([(source, s) for s in out_sents])
        backward = self._probs([(output, s) for s in src_sents])
        return {
            "nli_supported": sum(p[self.entail] for p in forward) / len(forward),
            "nli_contradicted": sum(max(range(len(p)), key=p.__getitem__) == self.contra for p in forward) / len(forward),
            "nli_coverage": sum(p[self.entail] for p in backward) / len(backward),
        }


# --- The pipeline's pass/fail rule -------------------------------------------------

def fkgl(text):
    return textstat.flesch_kincaid_grade(text) if text.strip() else None


def check_rewrite(source, rewrite, nli, coverage_min=NLI_COVERAGE_MIN):
    """Run the three checks. Returns {"passed": bool, "numbers": {...}, "negation": {...}, "nli": {...}}."""
    if not rewrite or not rewrite.strip():
        return {"passed": False, "numbers": {"passed": False}, "negation": {"passed": False},
                "nli": {"passed": False}, "reason": "empty rewrite"}
    src_nums, out_nums = numbers(source), numbers(rewrite)
    recall, precision = number_recall(source, rewrite), number_precision(source, rewrite)
    negation = negation_kept(source, rewrite)
    nli_scores = nli.score(source, rewrite)
    coverage = nli_scores["nli_coverage"]
    result = {
        "numbers": {"passed": recall in (None, 1.0) and precision in (None, 1.0), "recall": recall,
                    "precision": precision, "missing": sorted(src_nums - out_nums), "added": sorted(out_nums - src_nums)},
        "negation": {"passed": negation != 0.0, "kept": negation},
        "nli": {"passed": coverage is not None and coverage >= coverage_min, "threshold": coverage_min, **nli_scores},
        "fkgl": {"source": fkgl(source), "rewrite": fkgl(rewrite)},
    }
    result["passed"] = all(result[k]["passed"] for k in ("numbers", "negation", "nli"))
    return result
