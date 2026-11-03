"""The whole system: contract in, question in, grounded plain-English answer out.

    upload ──► 1 ingest + chunk ──► (per question) 2 route ──► 3 BM25 top 10
           ──► 4 DeBERTa span + score ──(below threshold)──► "Not enough evidence in this contract."
           ──► 5 expand to clause ──► 6 Gemma rewrite ──► 7 checks
           ──► answer = rewrite + original clause + section/page + confidence
               (checks failed: the original clause with a warning)

Conversation adds the "lightweight dialogue" from the proposal: it keeps the active
topic, asks which topic was meant when the router is torn between two, and answers
"anything else?" with the next-best passage for the active topic.
"""
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from .checks import NLIScorer, check_rewrite
from .chunking import chunk_text
from .config import CATEGORIES, CUAD_QUESTIONS, TOPIC_LABELS
from .expand import clause_span
from .extractor import Extractor
from .ingest import read_document
from .retrieval import ChunkIndex
from .router import GemmaRouter, RuleRouter, rule_hits

NO_EVIDENCE = "Not enough evidence in this contract."
CHECK_WARNING = ("The plain-English version did not pass the meaning checks, so here is the original "
                 "clause instead. Read it carefully, or ask a lawyer.")


@dataclass
class Answer:
    kind: str           # answer | unchecked | no_evidence | out_of_scope | clarify | help | error
    message: str
    category: str | None = None
    topic: str | None = None
    rewrite: str | None = None
    clause: str | None = None
    span: str | None = None
    highlight: list | None = None       # [start, end] of the span inside `clause`
    section: list | None = None         # heading path, outermost first
    pages: list | None = None           # pages of the evidence span
    confidence: dict | None = None      # {"score", "threshold", "probability"}
    checks: dict | None = None
    retrieved: list | None = None       # BM25 top 10: chunk, rank, score, pages
    route: dict | None = None
    options: list | None = None         # for clarify: [{"category", "topic"}]
    timings: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Contract:
    doc_id: str
    doc: object         # ingest.Document
    chunks: list
    index: ChunkIndex

    def summary(self):
        d = self.doc
        return {"doc_id": self.doc_id, "name": d.name, "kind": d.kind, "pages": d.n_pages,
                "words": len(d.text.split()), "chunks": len(self.chunks), "headings": len(d.headings),
                "warnings": d.warnings}


def open_contract(data, name):
    """Step 1: read the file, chunk it and index the chunks. Needs no models."""
    doc = read_document(data, name)
    doc_id = uuid.uuid4().hex[:12]
    chunks = chunk_text(doc.text, doc_id)
    return Contract(doc_id, doc, chunks, ChunkIndex(chunks))


def _highlight(clause, span):
    """Where the span sits inside the (whitespace-normalised) clause, or None."""
    norm = re.sub(r"\s*\n\s*", "\n", span).strip()
    i = clause.find(norm)
    if i >= 0:
        return [i, i + len(norm)]
    pattern = r"\s+".join(map(re.escape, span.split()))
    m = re.search(pattern, clause) if pattern else None
    return [m.start(), m.end()] if m else None


class Jargonaut:
    """Loads the models once and answers questions about any number of contracts."""

    def __init__(self, load_simplifier=True, log=print):
        t = time.time()
        log("Loading the NLI model (checks) ...")
        self.nli = NLIScorer()
        log("Loading DeBERTa ...")
        self.extractor = Extractor()
        self.simplifier = None
        if load_simplifier:
            from .simplifier import Simplifier

            log("Loading Gemma 4 E4B + adapter ...")
            self.simplifier = Simplifier()
        # Gemma (adapter off) routes questions; without it, keyword rules + NLI do.
        self.router = GemmaRouter(self.simplifier) if self.simplifier else RuleRouter(self.nli)
        self.lock = threading.Lock()   # one GPU: one request at a time
        log(f"Models ready in {time.time() - t:.0f} s.")

    # --- Steps 3-7 for a known category ------------------------------------------
    def answer_category(self, contract, category, route=None):
        timings, t = {}, time.time()
        question = CUAD_QUESTIONS[category]
        candidates = contract.index.top_k(question)
        timings["bm25"] = time.time() - t
        retrieved = [{"chunk_id": c.chunk_id, "rank": r, "bm25": round(s, 4),
                      "pages": contract.doc.pages(c.char_start, c.char_end)}
                     for r, (c, s) in enumerate(candidates, 1)]
        t = time.time()
        with self.lock:
            extraction = self.extractor.extract(question, candidates)
        timings["deberta"] = time.time() - t
        base = dict(category=category, topic=TOPIC_LABELS[category], retrieved=retrieved, route=route)
        if not extraction.accepted:
            best = extraction.best
            return Answer("no_evidence", NO_EVIDENCE, confidence=self._confidence(extraction, best),
                          timings=timings, **base), extraction
        return self.explain(contract, extraction.best, extraction, base, timings), extraction

    def explain(self, contract, span, extraction, base, timings=None):
        """Steps 5-7 for one accepted span."""
        timings = dict(timings or {})
        doc = contract.doc
        clause = (clause_span(doc.text, span.start, span.end) or (span.text.strip(),))[0]
        common = dict(clause=clause, span=span.text, highlight=_highlight(clause, span.text),
                      section=[h.text for h in doc.section(span.start)], pages=doc.pages(span.start, span.end),
                      confidence=self._confidence(extraction, span), **base)
        if self.simplifier is None:
            return Answer("unchecked", "Gemma is not loaded; showing the original clause.", timings=timings, **common)
        t = time.time()
        with self.lock:
            rewrite = self.simplifier.rewrite(clause)
            timings["gemma"] = time.time() - t
            t = time.time()
            checks = check_rewrite(clause, rewrite, self.nli)
        timings["checks"] = time.time() - t
        if checks["passed"]:
            return Answer("answer", rewrite, rewrite=rewrite, checks=checks, timings=timings, **common)
        return Answer("unchecked", CHECK_WARNING, rewrite=rewrite, checks=checks, timings=timings, **common)

    def _confidence(self, extraction, span):
        if span is None:
            return {"score": None, "threshold": extraction.threshold, "probability": None}
        return {"score": round(span.diff, 3), "threshold": extraction.threshold,
                "probability": self.extractor.probability(span.diff)}

    # --- Step 2 + the rest ------------------------------------------------------
    def route(self, question):
        with self.lock:
            return self.router.route(question)

    def ask(self, contract, question):
        """One question, no conversation state."""
        return Conversation(self, contract).reply(question)


# Whole-message matches only: "help me with the insurance clause" is a question, not a request for help.
_HELP = re.compile(r"^\s*(?:hi|hello|hey|help|\?+|what can (?:you|i) (?:do|ask)|what do you (?:do|know)|topics|menu)"
                   r"(?: please)?\s*[.!?]*\s*$", re.I)
_MORE = re.compile(r"^\s*(?:more|show (?:me )?more|tell me more|anything else|what else|any others?"
                   r"|other (?:clauses?|parts?|places?|ones?)|next|another(?: one)?|is there (?:more|anything else))"
                   r"(?: please)?\s*[.!?]*\s*$", re.I)
_ORDINALS = {"1": 0, "one": 0, "first": 0, "the first": 0, "the first one": 0, "2": 1, "two": 1, "second": 1,
             "the second": 1, "the second one": 1}


class Conversation:
    """Dialogue state for one contract: active topic, pending clarification, unseen passages."""

    def __init__(self, system, contract):
        self.system, self.contract = system, contract
        self.active = None          # category of the last answer
        self.pending = None         # categories offered in a clarification question
        self.alternatives = []      # further accepted spans for the active topic
        self.last_extraction = None

    def help(self):
        topics = ", ".join(TOPIC_LABELS[c].lower() for c in CATEGORIES)
        return Answer("help", f"Ask me about this contract. I can explain: {topics}.",
                      options=[{"category": c, "topic": TOPIC_LABELS[c]} for c in CATEGORIES])

    def reply(self, message):
        text = " ".join(message.split())
        if not text:
            return self.help()
        if self.pending:
            options, self.pending = self.pending, None
            choice = _ORDINALS.get(text.lower().rstrip(".!"))
            if choice is not None and choice < len(options):
                return self._answer(options[choice], {"method": "clarified"})
            picked = [c for c in rule_hits(text) if c in options] or [
                c for c in options if TOPIC_LABELS[c].lower() in text.lower() or c.lower() in text.lower()]
            if len(picked) == 1:
                return self._answer(picked[0], {"method": "clarified"})
        if _HELP.match(text):
            return self.help()
        if _MORE.match(text) and self.active:
            return self._more()
        t = time.time()
        route = self.system.route(text)
        route_info = {"method": route.method, "rule_hits": route.rule_hits,
                      "scores": {c: round(s, 3) for c, s in route.scores.items()},
                      "best_distractor": route.best_distractor, "seconds": round(time.time() - t, 3)}
        if route.category:
            return self._answer(route.category, route_info)
        if route.method == "clarify":
            self.pending = route.options
            a, b = (TOPIC_LABELS[c].lower() for c in route.options)
            return Answer("clarify", f"Do you mean (1) {a} or (2) {b}?", route=route_info,
                          options=[{"category": c, "topic": TOPIC_LABELS[c]} for c in route.options])
        topics = "; ".join(TOPIC_LABELS[c].lower() for c in CATEGORIES)
        return Answer("out_of_scope", "That question is outside what I can answer from this contract. "
                      f"I can explain these topics: {topics}.", route=route_info,
                      options=[{"category": c, "topic": TOPIC_LABELS[c]} for c in CATEGORIES])

    def reply_category(self, category):
        """Answer a topic chosen directly (a topic button), skipping the router."""
        self.pending = None
        return self._answer(category, {"method": "chosen"})

    def _answer(self, category, route_info):
        answer, extraction = self.system.answer_category(self.contract, category, route_info)
        self.active, self.last_extraction = category, extraction
        self.alternatives = list(extraction.alternatives) if extraction.accepted else []
        return answer

    def _more(self):
        topic = TOPIC_LABELS[self.active].lower()
        if not self.alternatives:
            return Answer("no_evidence", f"I found no other passage about {topic} that I am confident about.",
                          category=self.active, topic=TOPIC_LABELS[self.active])
        span = self.alternatives.pop(0)
        base = dict(category=self.active, topic=TOPIC_LABELS[self.active], route={"method": "more"})
        return self.system.explain(self.contract, span, self.last_extraction, base)
