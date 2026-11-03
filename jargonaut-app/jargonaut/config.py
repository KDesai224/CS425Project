"""Paths, model settings and thresholds for the whole pipeline, in one place.

Every setting here was fixed by the component it belongs to (data prep, BM25,
DeBERTa, Gemma), and the comment says where it comes from. Paths default to the
container mounts in docker-compose.yml; set the environment variables to run
anywhere else.
"""
import json
import os
from pathlib import Path

# --- Model locations ----------------------------------------------------------
# DeBERTa run folder: final/ (model + tokenizer) and threshold.json, from deberta-docker/runs/<run>/
DEBERTA_RUN = Path(os.environ.get("DEBERTA_RUN", "/models/deberta"))
# Gemma LoRA adapter: Run 6 (e4b-r16-lr0.0002-e2-s42-syn), selected checkpoint
ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", "/models/adapter"))
GEMMA_ID = os.environ.get("GEMMA_ID", "google/gemma-4-E4B-it")
# The NLI model Grpproj/jargonaut/metrics.py scores faithfulness with
NLI_ID = os.environ.get("NLI_ID", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
# Optional: Platt scaling of the DeBERTa score, written by scripts/calibrate.py
CALIBRATION = Path(os.environ.get("CALIBRATION", Path(__file__).resolve().parent.parent / "eval" / "calibration.json"))

# --- The 10 categories --------------------------------------------------------
# The categories proposed in data_prep/README.md, section 1, and used by BM25 and
# DeBERTa. Each has exactly one CUAD question. The Details strings are copied
# character for character from qa.jsonl.gz, including the non-breaking spaces,
# because DeBERTa was trained on these exact strings (scripts/check_parity.py checks them).
CUAD_TEMPLATE = ('Highlight the parts (if any) of this contract related to "{category}" '
                 'that should be reviewed by a lawyer. Details: {details}')
CUAD_DETAILS = {
    "Governing Law": "Which state/country's law governs the interpretation of the contract?",
    "Expiration Date": "On what date will the contract's initial term expire?",
    "Renewal Term": "What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.",
    "Termination For Convenience": "Can a party terminate this\xa0 contract without cause (solely by giving a notice and allowing a waiting\xa0 period to expire)?",
    "Cap On Liability": "Does the contract include a cap on liability upon the breach of a party’s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
    "Anti-Assignment": "Is consent or notice required of a party if the contract is assigned to a third party?",
    "Non-Compete": "Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?\xa0",
    "Exclusivity": "Is there an exclusive dealing\xa0 commitment with the counterparty? This includes a commitment to procure all “requirements” from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on\xa0 collaborating or working with other parties), whether during the contract or\xa0 after the contract ends (or both).",
    "Ip Ownership Assignment": "Does intellectual property created\xa0 by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?",
    "Insurance": "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?",
}
CATEGORIES = list(CUAD_DETAILS)
CUAD_QUESTIONS = {c: CUAD_TEMPLATE.format(category=c, details=d) for c, d in CUAD_DETAILS.items()}

# How each category is named to the user
TOPIC_LABELS = {
    "Governing Law": "Governing law",
    "Expiration Date": "When the contract ends",
    "Renewal Term": "Renewal",
    "Termination For Convenience": "Ending the contract without a reason",
    "Cap On Liability": "Limits on liability",
    "Anti-Assignment": "Transferring the contract to someone else",
    "Non-Compete": "Restrictions on competing",
    "Exclusivity": "Exclusive dealing",
    "Ip Ownership Assignment": "Who owns intellectual property",
    "Insurance": "Insurance requirements",
}

# --- Step 1: chunking (data_prep/outputs/chunks/chunks_words_m300_o60/chunk_config.json) ---
CHUNK_MAX_WORDS = 300
CHUNK_OVERLAP = 60

# --- Step 3: BM25 (BM25_Retrieval_and_Evaluation.ipynb, step 5 "FINAL") ------
BM25_TOP_K = 10
BM25_K1 = 1.5
BM25_B = 0.75

# --- Step 4: DeBERTa (deBERTa_training.ipynb; batch size from deberta-docker) --
QA_MAX_LEN = 512
QA_STRIDE = 128
QA_TOPK = 20          # start/end logits searched per window
QA_BATCH_SIZE = 16


def deberta_settings():
    """The tuned answerability threshold and answer-length cap saved with the run."""
    with open(DEBERTA_RUN / "threshold.json", encoding="utf-8") as f:
        saved = json.load(f)
    return {"threshold": float(saved["threshold"]), "max_answer_tokens": int(saved["max_answer_tokens"])}


# --- Step 5: clause expansion (Grpproj/scripts/02_build_pipeline_set.py) -----
CLAUSE_MIN_WORDS = 40
CLAUSE_MAX_WORDS = 300

# --- Step 6: Gemma (Grpproj/configs/e4b.yaml, generate:) ---------------------
GEMMA_MAX_NEW_TOKENS = 512
GEMMA_LOAD_IN_4BIT = True

# --- Step 7: checks ----------------------------------------------------------
# Same bar as the "synthetic strict" training filter in Grpproj step 10.
NLI_COVERAGE_MIN = float(os.environ.get("NLI_COVERAGE_MIN", 0.7))
