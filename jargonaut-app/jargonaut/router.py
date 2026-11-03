"""Step 2: map a free-text question to one of the 10 categories, or to "out of scope".

Primary router (GemmaRouter): the Gemma model that writes the rewrites, with its
LoRA adapter switched off, answers a multiple-choice prompt: the 10 topics plus
"none of these", each described in plain words and with CUAD's own boundaries
(a liability cap is not an indemnity or a penalty, a non-compete is not a ban on
hiring staff). One forward pass gives a probability for every option, read off
the next-token logits of the option letters, so nothing is generated or parsed.
  - "none of these" wins                      -> out of scope
  - two topics both plausible (runner-up >= CLARIFY_AT) -> ask which one was meant
  - otherwise                                 -> the best topic

Fallback (RuleRouter), used when Gemma is not loaded: keyword rules, then zero-shot
NLI with the checks' model. It is weaker; scripts/eval_router.py reports both.

The option wording and CLARIFY_AT were chosen on the dev half of
eval/router_questions.jsonl; the test half was run once afterwards.
"""
import re
from dataclasses import dataclass, field

from .config import CATEGORIES

OOS = "OUT_OF_SCOPE"


@dataclass
class Route:
    category: str | None          # None = out of scope, or a clarification is needed
    method: str                   # "gemma", "rule", "rule+nli", "nli", "out_of_scope", "clarify"
    rule_hits: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)       # probability (Gemma) or entailment (NLI) per option
    options: list = field(default_factory=list)      # for "clarify": the two candidate categories
    best_distractor: tuple | None = None


# --- Primary: Gemma multiple choice ---------------------------------------------

CLARIFY_AT = 0.25
GEMMA_OPTIONS = [
    ("Governing Law", "Governing law and jurisdiction: which state's or country's law applies to the contract, or which courts handle disputes"),
    ("Expiration Date", "Expiration date: when the contract's term ends or how long it lasts"),
    ("Renewal Term", "Renewal term: whether the contract renews or is extended after its term, automatically or by notice"),
    ("Termination For Convenience", "Termination for convenience: whether a party can end the contract early without a reason, just by giving notice"),
    ("Cap On Liability", "Cap on liability: the most a party can be made to pay for losses or damages, or time limits for bringing claims "
                         "(not fixed penalties and not indemnities)"),
    ("Anti-Assignment", "Assignment: whether the contract can be assigned or transferred to someone else, and whose consent that needs"),
    ("Non-Compete", "Non-compete: restrictions on competing with the other party or working in a business, field or area "
                    "(not hiring the other party's staff)"),
    ("Exclusivity", "Exclusivity: exclusive rights or territories, or having to buy from, sell to or work with only one party"),
    ("Ip Ownership Assignment", "IP ownership: who owns, or must assign, inventions, work product or intellectual property created under the contract"),
    ("Insurance", "Insurance: insurance a party must carry, its coverage amounts, and who must be named as insured"),
    (OOS, "None of these: anything else, such as payments, confidentiality, warranties, indemnities (covering the other side's "
          "costs when someone sues), liquidated damages or penalties, hiring or soliciting staff, a change of control, dispute "
          "procedures, ending the contract because of a breach, or questions unrelated to the contract"),
]
_LETTERS = "ABCDEFGHIJK"
_ROUTER_SYSTEM = "You sort questions that people ask about a contract they have uploaded into topics."


def gemma_messages(question):
    lines = "\n".join(f"{letter}. {text}" for letter, (_, text) in zip(_LETTERS, GEMMA_OPTIONS))
    return [{"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nWhich topic is this question about?\n{lines}\n\n"
                                        "Answer with the letter only."}]


def decide_gemma(probs, clarify_at=CLARIFY_AT):
    """Route from the option probabilities ({category or OOS: p})."""
    top, second = sorted(probs, key=probs.get, reverse=True)[:2]
    if top == OOS:
        return Route(None, "out_of_scope", scores=probs)
    if second != OOS and probs[second] >= clarify_at:
        return Route(None, "clarify", scores=probs, options=[top, second])
    return Route(top, "gemma", scores=probs)


class GemmaRouter:
    def __init__(self, simplifier, clarify_at=CLARIFY_AT):
        """simplifier: a simplifier.Simplifier; its model is used with the adapter switched off."""
        self.tokenizer, self.model, self.torch = simplifier.tokenizer, simplifier.model, simplifier.torch
        self.clarify_at = clarify_at
        ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in _LETTERS]
        assert all(len(i) == 1 for i in ids), "each option letter must be one token"
        self.letter_ids = [i[0] for i in ids]

    def probabilities(self, question):
        torch = self.torch
        text = self.tokenizer.apply_chat_template(gemma_messages(question), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        adapter_off = self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else _NoAdapter()
        with adapter_off, torch.inference_mode():
            logits = self.model(**enc).logits[0, -1].float()
        p = torch.softmax(logits[self.letter_ids], -1).tolist()
        return {key: prob for (key, _), prob in zip(GEMMA_OPTIONS, p)}

    def route(self, question):
        route = decide_gemma(self.probabilities(question), self.clarify_at)
        route.rule_hits = rule_hits(question)     # shown for transparency only
        return route


class _NoAdapter:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- Fallback: keyword rules, then zero-shot NLI -----------------------------------

# The question needs this much entailment probability for a category to be accepted,
# and a runner-up within MARGIN of the winner (also above THRESHOLD) triggers a
# clarification question.
THRESHOLD = 0.5
MARGIN = 0.05

_I = re.I
_PARTY = (r"(?:i|we|they|you|either\s+(?:party|side)|the\s+other\s+(?:side|party)|one\s+party|a\s+party|both\s+part(?:y|ies)"
          r"|the\s+(?:customer|client|supplier|vendor|company|licensee|licensor|distributor|contractor|provider|buyer|seller))")
_CONTRACT = r"(?:the|this|my|our|their|that)\s+(?:contract|agreement|deal)"
_IP_WORDS = (r"\b(?:intellectual\s+property|ip\b|patents?|copyrights?|invent\w*|work\s+product|deliverables|designs?"
             r"|source\s+code|software|code\s+(?:i|we)\s+wr|know-?how|trade\s+secrets?|works?\s+(?:made\s+)?for\s+hire)")

RULES = {
    "Governing Law": [
        r"\bgovern\w*\b.{0,40}\blaws?\b", r"\blaws?\b.{0,40}\bgovern", r"\bjurisdiction",
        r"\b(?:which|what|whose)\s+(?:state|country|countries|nation)(?:'s|s)?\b.{0,20}\blaws?\b",
        r"\b(?:which|what|whose)\s+laws?\b", r"\bchoice\s+of\s+law", r"\bapplicable\s+law",
        r"\b(?:which|what)\s+courts?\b", r"\bunder\s+(?:which|what)\s+law",
        r"\blaws?\s+of\s+(?:the\s+)?(?:state|country|england|new\s+york|delaware|california|texas|singapore)",
    ],
    "Expiration Date": [
        r"\bexpir\w*", r"\bend\s+date\b", r"\bend\s+of\s+the\s+term\b",
        r"\bwhen\s+(?:does|will|do|is)\s+(?:it|this|" + _CONTRACT + r"|the\s+term)\s+(?:end|finish|stop|run\s+out|over|done|up)\b",
        r"\bhow\s+long\b.{0,40}\b(?:last|valid|in\s+(?:force|effect)|run|bound|stuck|locked|contract|agreement|deal|term)\b",
        r"\b(?:duration|length)\s+of\s+(?:the\s+|this\s+)?(?:contract|agreement|term|deal)", r"\binitial\s+term\b",
        r"\b(?:valid|in\s+effect|in\s+force|good|run|runs|last|lasts)\s+until\b", r"\buntil\s+when\b",
    ],
    "Renewal Term": [
        r"\brenew\w*", r"\broll(?:s|ed|ing)?\s+over", r"\bevergreen",
        r"\bauto(?:matic(?:ally)?)?[-\s]*(?:renew|extend|continu)\w*",
        r"\bextend\w*\b.{0,40}\b(?:term|contract|agreement|deal)\b", r"\bextension\b",
    ],
    "Termination For Convenience": [
        r"\bterminat\w*\s+(?:for\s+convenience|without\s+(?:cause|a\s+reason|any\s+reason|reason)|at\s+will|for\s+any\s+reason|for\s+no\s+reason)",
        r"\bfor\s+convenience\b",
        r"\b(?:can|could|may|am|are|is|do|does)\s+" + _PARTY + r"\b.{0,30}\b(?:terminat\w*|cancel\w*|end|exit|quit|leave|walk\s+away"
        r"|back\s+out|pull\s+out|opt\s+out|get\s+out|break|withdraw)\b",
        r"\bget\s+out\s+of\s+" + _CONTRACT, r"\bearly\s+terminat\w*", r"\bterminat\w*\s+early\b",
        r"\bcancel\w*\s+" + _CONTRACT, r"\bexit\s+clause\b", r"\bout\s+clause\b",
    ],
    "Cap On Liability": [
        r"\bliabilit\w*", r"\bliable\b", r"\bcap(?:ped|s)?\s+(?:on|at)\b", r"\bcapped\b",
        r"\bmaximum\s+(?:amount|damages|liability|payout|exposure)", r"\blimit\w*\s+(?:of|on)\s+(?:liability|damages)",
        r"\bhow\s+much\b.{0,30}\b(?:sued|sue|owe|liable|on\s+the\s+hook|damages|compensat\w*|recover)",
        r"\bon\s+the\s+hook\b", r"\bconsequential\b",
    ],
    "Anti-Assignment": [
        r"\bassign(?:ment|ed|ing|s)?\b", r"\bnovat\w*",
        r"\btransfer\w*\b.{0,40}\b(?:contract|agreement|deal|rights|obligations)\b",
        r"\b(?:hand|pass)\w*\s+(?:it|this|the\s+(?:contract|agreement|deal))\s+(?:over|on)\b",
    ],
    "Non-Compete": [
        r"\bcompet\w*", r"\bnon-?compete", r"\brestrictive\s+covenant", r"\brivals?\b",
        r"\bsame\s+(?:industry|line\s+of\s+business|market|field|sector|space)\b",
    ],
    "Exclusivity": [
        r"\bexclusiv\w*", r"\bsole\s+(?:supplier|source|distributor|provider|vendor|customer|partner|reseller)",
        r"\bonly\s+(?:buy|purchase|source|order|use|sell|work|deal)\b",
        r"\b(?:other|another|different|alternative|outside|competing)\s+(?:suppliers?|vendors?|providers?|distributors?|resellers?)",
    ],
    "Ip Ownership Assignment": [
        _IP_WORDS + r".{0,60}\b(?:own|owns|owned|owner\w*|belong\w*|assign\w*|title|rights?\s+(?:to|in|over)|keep)\b",
        r"\b(?:own|owns|owned|owner\w*|belong\w*|title\s+to|rights?\s+(?:to|in|over))\b.{0,60}" + _IP_WORDS,
        r"\bwork(?:s)?\s+(?:made\s+)?for\s+hire\b", r"\bintellectual\s+property\b",
    ],
    "Insurance": [
        r"\binsur\w*", r"\bpolicy\s+limits?\b", r"\bcertificates?\s+of\s+(?:insurance|coverage)",
        r"\b(?:umbrella|workers'?\s+comp\w*|errors\s+and\s+omissions)\b",
        r"\b(?:liability|umbrella|indemnity)\s+(?:policy|policies|cover|coverage)\b",
        r"\bpolic(?:y|ies)\b.{0,30}\bcovers?\b",
    ],
}
# A rule hit is ignored when the question also matches its veto.
VETOES = {
    "Termination For Convenience": r"\b(?:breach\w*|default\w*|violat\w*|insolven\w*|bankrupt\w*|(?:for|with)\s+cause|fails?\s+to"
                                   r"|failure|missed|late|sloppy|doesn'?t\s+(?:pay|deliver|perform)|stops?\s+paying|misconduct"
                                   r"|force\s+majeure|change\s+of\s+control|bought|acquired)\b",
    "Cap On Liability": r"\bindemn\w*|\bliquidated\b|\bpenalt\w*|\binsur\w*|\bpolic(?:y|ies)\b|\bcoverage\b|\bwarrant\w*",
    "Exclusivity": r"\bjurisdiction|\bcourts?\b",
    "Anti-Assignment": _IP_WORDS + r"|\bchange\s+of\s+control\b",
    "Non-Compete": r"\bsolicit\w*|\bpoach\w*|\bhir(?:e|ing)\b|\bcompetitive\s+pric\w*|\bbought\b",
    "Expiration Date": r"\brenew\w*|\bwarrant\w*|\bconfidential\w*",
    "Governing Law": r"\barbitrat\w*|\bmediat\w*",
}
_COMPILED = {c: [re.compile(p, _I) for p in ps] for c, ps in RULES.items()}
_VETO = {c: re.compile(p, _I) for c, p in VETOES.items()}

# NLI hypotheses: "This question is about <phrase>."
HYPOTHESES = {
    "Governing Law": "which law or which courts govern the contract",
    "Expiration Date": "when the contract expires or how long it lasts",
    "Renewal Term": "whether the contract renews or can be extended",
    "Termination For Convenience": "ending the contract early without having to give a reason",
    "Cap On Liability": "limits on how much a party must pay if something goes wrong",
    "Anti-Assignment": "transferring the contract to another company or person",
    "Non-Compete": "restrictions on competing with the other party or working in the same business",
    "Exclusivity": "having to deal exclusively with one party",
    "Ip Ownership Assignment": "who owns intellectual property or work created under the contract",
    "Insurance": "insurance that a party must have",
}
DISTRACTORS = [
    "payments, fees, prices or invoices", "keeping information confidential", "who the parties to the contract are",
    "when the contract starts or was signed", "warranties or guarantees", "indemnifying the other party against claims",
    "resolving disputes through arbitration or mediation", "ending the contract because the other party broke it",
    "events beyond anyone's control such as disasters", "minimum amounts that must be purchased",
    "protecting personal data or privacy", "hiring away the other party's employees",
    "how to send notices to the other party", "audits or inspecting records", "penalties or liquidated damages",
    "a license to use technology, software or trademarks", "something that has nothing to do with a contract",
]
_TEMPLATE = "This question is about {}."


def rule_hits(question):
    hits = []
    for category, patterns in _COMPILED.items():
        if any(p.search(question) for p in patterns):
            veto = _VETO.get(category)
            if not (veto and veto.search(question)):
                hits.append(category)
    return hits


def decide(hits, scores, distractors, threshold=THRESHOLD, margin=MARGIN):
    """The fallback's decision from rule hits and NLI scores (separate so thresholds can be swept)."""
    best_distractor = max(distractors.items(), key=lambda kv: kv[1]) if distractors else None
    if len(hits) == 1:
        return Route(hits[0], "rule", hits, scores, best_distractor=best_distractor)
    if hits:
        return Route(max(hits, key=scores.get), "rule+nli", hits, scores, best_distractor=best_distractor)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top, second = ranked[0], ranked[1]
    if scores[top] < threshold or (best_distractor and best_distractor[1] > scores[top]):
        return Route(None, "out_of_scope", hits, scores, best_distractor=best_distractor)
    if scores[second] >= threshold and scores[top] - scores[second] < margin:
        return Route(None, "clarify", hits, scores, options=[top, second], best_distractor=best_distractor)
    return Route(top, "nli", hits, scores, best_distractor=best_distractor)


class RuleRouter:
    def __init__(self, nli=None, threshold=THRESHOLD, margin=MARGIN):
        """nli: a checks.NLIScorer, whose model is shared. Without it only the rules are used."""
        self.nli, self.threshold, self.margin = nli, threshold, margin

    def nli_scores(self, question):
        labels = [(c, HYPOTHESES[c]) for c in CATEGORIES] + [(None, d) for d in DISTRACTORS]
        probs = self.nli._probs([(question, _TEMPLATE.format(h)) for _, h in labels])
        cats = {c: p[self.nli.entail] for (c, _), p in zip(labels, probs) if c}
        distractors = {h: p[self.nli.entail] for (c, h), p in zip(labels, probs) if c is None}
        return cats, distractors

    def route(self, question):
        hits = rule_hits(question)
        if self.nli is None:
            if len(hits) == 1:
                return Route(hits[0], "rule", hits)
            return Route(None, "out_of_scope" if not hits else "clarify", hits, options=hits[:2])
        scores, distractors = self.nli_scores(question)
        return decide(hits, scores, distractors, self.threshold, self.margin)
