"""Chunking, clause expansion, BM25 tokenisation, routing rules, checks and highlighting (no models needed)."""
import re

from jargonaut.checks import check_rewrite, negation_kept, number_recall, numbers
from jargonaut.chunking import chunk_spans, chunk_text
from jargonaut.config import CATEGORIES, CUAD_QUESTIONS
from jargonaut.expand import clause_span, expand_to_clause
from jargonaut.pipeline import _highlight
from jargonaut.retrieval import ChunkIndex, query_form_b, tokenize
from jargonaut.router import OOS, decide_gemma, rule_hits


def test_chunk_windows_overlap_by_60_words():
    text = " ".join(f"w{i}" for i in range(700))
    spans = chunk_spans(text)
    words = [text[s:e].split() for s, e in spans]
    assert [len(w) for w in words] == [300, 300, 220]
    assert words[0][240:] == words[1][:60] and words[1][-1] == "w539" and words[2][-1] == "w699"
    assert chunk_text(text, "doc")[1].chunk_id == "doc_c0001"


def test_clause_span_matches_expand_to_clause_in_both_branches():
    short = "Heading\nThe term is two years.\nIt renews for one year unless a party gives notice.\n" + "Other text here. " * 10
    long_par = "Preamble words. " * 200 + "The Licensee shall not assign this Agreement. " + "More words follow here. " * 200
    for text, needle in [(short, "renews for one year"), (long_par, "shall not assign")]:
        s = text.index(needle)
        e = s + len(needle)
        passage, ps, pe = clause_span(text, s, e)
        assert passage == expand_to_clause(text, s, e)
        assert re.sub(r"\s+", " ", text[ps:pe]) == re.sub(r"\s+", " ", passage)
        assert len(passage.split()) <= 310


def test_bm25_tokeniser_keeps_negations_and_stems():
    tokens = tokenize("The Licensee shall not assign without consent")
    assert "the" not in tokens and "not" in tokens and "without" in tokens
    assert tokenize("assignment") == tokenize("assign") == ["assign"]
    q = query_form_b(CUAD_QUESTIONS["Governing Law"])
    assert q.startswith("Governing Law Which state/country's law") and "Highlight" not in q


def test_bm25_ranks_the_matching_chunk_first():
    # Long enough for realistic IDF: the clause sits in 1-2 of about 12 chunks
    chunks = chunk_text(" ".join(["filler words about payment schedules and invoices"] * 200)
                        + " This Agreement shall be governed by the laws of the State of New York. "
                        + " ".join(["more filler about delivery logistics and shipping"] * 200), "c")
    top = ChunkIndex(chunks).top_k(CUAD_QUESTIONS["Governing Law"], k=3)
    assert "governed by the laws" in top[0][0].text


def test_router_rules_and_vetoes():
    assert rule_hits("Do they need to carry insurance?") == ["Insurance"]
    assert rule_hits("Which law governs this agreement?") == ["Governing Law"]
    assert "Termination For Convenience" in rule_hits("Can I cancel the contract whenever I want?")
    assert "Termination For Convenience" not in rule_hits("If they breach, can I terminate the contract?")
    assert "Exclusivity" not in rule_hits("Do the courts of Delaware have exclusive jurisdiction?")
    assert "Cap On Liability" not in rule_hits("Do I have to indemnify them for third-party claims and liability?")


def test_gemma_decision():
    p = dict.fromkeys(CATEGORIES + [OOS], 0.0)
    assert decide_gemma({**p, "Insurance": 0.9, OOS: 0.1}).category == "Insurance"
    assert decide_gemma({**p, OOS: 0.6, "Insurance": 0.4}).method == "out_of_scope"
    r = decide_gemma({**p, "Non-Compete": 0.6, "Exclusivity": 0.4})
    assert r.method == "clarify" and r.options == ["Non-Compete", "Exclusivity"]


class FakeNLI:
    def __init__(self, coverage):
        self.coverage = coverage

    def score(self, source, output):
        return {"nli_supported": 0.9, "nli_contradicted": 0.0, "nli_coverage": self.coverage}


CLAUSE = ("Either party may terminate this Agreement upon thirty (30) days written notice, "
          "except that the Licensee may not terminate during the first twelve (12) months.")


def test_checks_pass_and_fail():
    good = "Either side can end the deal with 30 days' written notice. But the Licensee cannot end it in the first 12 months."
    assert numbers(CLAUSE) == {"30", "12"} and number_recall(CLAUSE, good) == 1.0
    assert check_rewrite(CLAUSE, good, FakeNLI(0.8))["passed"]
    dropped = "Either side can end the deal with written notice. But the Licensee cannot end it in the first 12 months."
    r = check_rewrite(CLAUSE, dropped, FakeNLI(0.8))
    assert not r["passed"] and r["numbers"]["missing"] == ["30"]
    flipped = "Either side can end the deal with 30 days' written notice, and the Licensee can end it in the first 12 months."
    assert negation_kept(CLAUSE, flipped) == 0.0 and not check_rewrite(CLAUSE, flipped, FakeNLI(0.8))["passed"]
    assert not check_rewrite(CLAUSE, good, FakeNLI(0.5))["passed"]          # NLI coverage below 0.7
    assert not check_rewrite(CLAUSE, "", FakeNLI(0.9))["passed"]


def test_highlight_finds_the_span_inside_the_clause():
    clause = "Section 9.2\nEither party may terminate this Agreement\nupon notice."
    assert clause[slice(*_highlight(clause, "terminate this Agreement upon"))] == "terminate this Agreement\nupon"
    assert _highlight(clause, "not in the clause") is None
