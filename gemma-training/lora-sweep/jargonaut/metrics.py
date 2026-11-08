"""Evaluation metrics for the simplifier.

Readability   : Flesch-Kincaid grade level, compression ratio
Simplification: SARI (needs a reference rewrite)
Faithfulness  : rule-based number/negation checks and two-way NLI entailment
Statistics    : bootstrap confidence intervals
"""
import random
import re
from collections import Counter

import textstat
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

_tok13a = Tokenizer13a()


# --- Readability -----------------------------------------------------------

def fkgl(text):
    return textstat.flesch_kincaid_grade(text) if text.strip() else None


def compression(source, output):
    """Output length over source length, in words. Well below 1 can mean dropped content."""
    return len(output.split()) / max(1, len(source.split()))


# --- SARI ------------------------------------------------------------------
# Port of the reference implementation (Xu et al. 2016) as used by Hugging Face
# `evaluate`: lowercase, 13a tokenisation, n-grams of order 1-4.

def _ngrams(tokens, n):
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _sari_ngram(sgrams, cgrams, rgramslist, numref):
    rcount = Counter(g for rgrams in rgramslist for g in rgrams)
    scount = Counter(sgrams)
    ccount = Counter(cgrams)
    srep = Counter({g: c * numref for g, c in scount.items()})
    crep = Counter({g: c * numref for g, c in ccount.items()})

    keep = srep & crep
    keep_good = keep & rcount
    keep_all = srep & rcount
    keep_p = keep_r = 1
    if keep:
        keep_p = sum(keep_good[g] / keep[g] for g in keep_good) / len(keep)
    if keep_all:
        keep_r = sum(keep_good.values()) / sum(keep_all.values())
    keep_f = 2 * keep_p * keep_r / (keep_p + keep_r) if keep_p > 0 or keep_r > 0 else 0

    dele = srep - crep
    dele_good = dele - rcount
    del_p = 1
    if dele:
        del_p = sum(dele_good[g] / dele[g] for g in dele_good) / len(dele)

    add = set(ccount) - set(scount)
    add_good = add & set(rcount)
    add_all = set(rcount) - set(scount)
    add_p = add_r = 1
    if add:
        add_p = len(add_good) / len(add)
    if add_all:
        add_r = len(add_good) / len(add_all)
    add_f = 2 * add_p * add_r / (add_p + add_r) if add_p > 0 or add_r > 0 else 0
    return keep_f, del_p, add_f


def sari(source, prediction, references):
    """SARI (0-100) for one item. `references` is a list of reference rewrites."""
    norm = lambda s: _tok13a(s.lower()).split()
    s, c, refs = norm(source), norm(prediction), [norm(r) for r in references]
    keep = dele = add = 0
    for n in range(1, 5):
        k, d, a = _sari_ngram(_ngrams(s, n), _ngrams(c, n), [_ngrams(r, n) for r in refs], len(refs))
        keep, dele, add = keep + k, dele + d, add + a
    return 100 * (keep + dele + add) / 12


# --- Rule-based faithfulness -------------------------------------------------

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


# --- NLI faithfulness --------------------------------------------------------

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

    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", batch_size=8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail, self.contra = labels["entailment"], labels["contradiction"]
        self.batch_size = batch_size

    def _probs(self, pairs):
        probs = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                return_tensors="pt", padding=True, truncation="only_first", max_length=1024,
            ).to(self.device)
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


# --- Statistics --------------------------------------------------------------

def bootstrap_ci(values, n_resamples=2000, seed=0):
    """Mean and 95% bootstrap confidence interval, ignoring None values."""
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_resamples))
    return sum(values) / len(values), means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def paired_bootstrap(a, b, n_resamples=2000, seed=0):
    """Mean of (a - b) over paired items with a 95% interval; the difference is
    significant at the 5% level when the interval excludes zero."""
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    return bootstrap_ci(diffs, n_resamples, seed)
