# Jargonaut app: findings

This is what was measured on the app in this folder, the whole pipeline
described in [README.md](README.md). It covers five things:

- whether the reused components are unchanged
- how well questions are routed
- whether the confidence score can be trusted
- how the whole pipeline does on CUAD's own PDF files of the test contracts
- how it does on the team's fictional test contracts (the `test` branch)

Every threshold was fixed before the test runs.

## Summary

- **The reused components are unchanged.** All six parity checks give the
  team's saved outputs back exactly, including DeBERTa's 1,020 test answers.
- **Routing works.** Gemma, with its adapter off, routes 97.3% of 110 held-out
  test questions correctly. No in-scope question goes to a wrong topic, and
  28 of 30 out-of-scope questions are rejected. The keyword + NLI fallback
  gets 76.4%.
- **The confidence score is calibrated.** After Platt scaling on validation,
  the expected calibration error on test is 0.039, and 95.3% of accepted
  answers are real evidence. The 0.6–0.8 band is overconfident.
- **Reading the PDF costs nothing measurable.** On the 102 test PDFs, the best
  span is on gold evidence for 81.9% of answerable questions, against 82.2%
  on CUAD's own text. Every accepted answer has a section and a page.
- **About half of the rewrites are shown.** 53.7% pass all three checks, and
  the reading grade falls from 18.9 to 8.3. NLI coverage holds back most of
  the rest.
- **The held-back rewrites are mostly false alarms.** In a reading of 40
  answers, 3 of the 16 held back had a real error (one reversed a right) and
  10 were false alarms. None of the 10 shown rewrites read in full changed
  the meaning.
- **The evaluation found PDF furniture, and it was fixed.** Page numbers and
  EDGAR "Source:" lines left in mid-page made the numbers check fail on
  numbers that were not in the clause. Fixing the reader raised that check
  from 86.8% to 93.7%.
- **On the team's fictional contracts, nothing false is accepted, but plainly
  worded governing-law clauses are missed.** Over 100 contract × topic pairs,
  precision is 19/19 and no trap sentence is accepted, and each contract's own
  clause is found 10 times out of 10. Recall is 19/28, because DeBERTa finds "New
  York law governs." but scores it below the threshold. Reworded as "This
  Agreement is governed by New York law.", all 10 are accepted. CUAD's training
  clauses almost never use the shorter form.

## The reused parts are unchanged

`scripts/check_parity.py` re-runs each component against the team's own saved
outputs. All six checks pass:

| Check | Result |
|---|---|
| The 10 CUAD questions in `config.py` vs `qa.jsonl.gz` | 5,100 rows, 0 mismatches |
| Chunking every contract vs Theo's `chunks.jsonl.gz` | 510 contracts, 16,824 chunks, all identical |
| BM25 top 10 vs `bm25_top10_{train,val,test}.jsonl.gz` | 5,090 questions, same chunks in the same order, same scores to 4 decimals |
| DeBERTa vs the local run's `test_preds.jsonl` | 1,020 test questions, 0 accept/abstain disagreements, 0 different spans, largest score difference 0.0000 |
| Copied Grpproj code (prompt, output cleanup, number/negation/NLI checks, clause expansion) | 20 definitions, character-identical |
| Clause expansion with offsets vs Grpproj's own function | 4,260 gold spans, same passage every time |

```powershell
docker compose run --rm jargonaut-app python scripts/check_parity.py      # about 7 minutes
```

## Routing (step 2)

CUAD has no user questions, so a separate agent wrote 220 of them from the 10
category descriptions alone, without seeing the router
(`eval/router_questions.jsonl`):

- **In scope:** 16 per topic, written the way non-lawyers type, many avoiding
  the topic's keyword.
- **Out of scope:** 60 questions. 40 are near-misses about other contract
  topics (late fees, indemnities, termination for breach, non-solicitation…)
  and 20 are unrelated to the contract.
- **Split:** 110 dev / 110 test. The router was designed on dev and run on
  test once.

| Router | Dev accuracy | Test accuracy | Test: in-scope routed right | Test: out-of-scope rejected |
|---|---|---|---|---|
| **Gemma multiple choice (the app's)** | 98.2% | **97.3%** | 79 / 80 (none to a wrong topic) | 28 / 30 |
| Keyword rules + zero-shot NLI (fallback without Gemma) | 83.6% | 76.4% | 57 / 80 (22 wrongly rejected) | 27 / 30 |

- **Methods tried on dev:** zero-shot NLI with three hypothesis wordings (61–76%), rules + NLI (82%), and Gemma multiple choice (93% at first, 98% after one round of option wording on dev).
- **Why NLI routing is weak:** questions are not statements, so the NLI model gave "which state's courts would I have to go to?" a Governing Law entailment of 0.002.
- **Cost:** routing with Gemma takes about 0.2 s and no extra memory.
- **The 3 test errors:** a deadline for bringing claims went to out of scope (CUAD counts claim time limits as Cap On Liability). "Can I give notice by email?" went to termination. A most-favoured-customer pricing question went to exclusivity.

```powershell
docker compose run --rm jargonaut-app python scripts/eval_router.py --split test --errors
docker compose run --rm jargonaut-app python scripts/eval_router.py --split test --fallback
```

## Confidence (step 4)

DeBERTa's score is a logit margin (accepted at ≥ 6.25), which a user cannot
read. `scripts/calibrate.py` fits Platt scaling on the validation split, the
split the threshold was tuned on. The fitted value is the probability that the
best span overlaps gold evidence, with unanswerable questions counted as misses.
The mapping is monotone, so no accept/abstain decision changes.

- **Fit:** P = 1 / (1 + e^-(0.326·score − 1.012)), which gives 0.74 at the threshold.
- **On test:** expected calibration error 0.039, Brier score 0.045.
- **Accepted answers on test:** 95.3% are real evidence, against a mean predicted 92.8%.
- **Caveat:** the 0.6–0.8 band is overconfident (predicted 0.71, observed 0.47 on 32 test questions). The UI therefore calls only ≥ 0.8 "High".

```powershell
docker compose run --rm jargonaut-app python scripts/calibrate.py         # writes eval/calibration.json
```

## End to end on the CUAD test contracts' PDFs

`scripts/eval_end_to_end.py` runs steps 1 and 3–7 on CUAD's own PDF files of the
102 test contracts, read the way an upload is read, for all 10 categories. That
is 1,020 questions, 443 of them with gold evidence. Routing is left out, since
CUAD has no user questions (see above). Gold spans are offsets into CUAD's
text, so they are found again in the app's text by matching letters and digits
only. No threshold was tuned on this run.

### Steps 1 to 5

| Step | Measure | Result |
|---|---|---|
| 1 Ingest | Words, relative to CUAD's text of the same contract | 98.9% (CUAD's text keeps the page numbers of 49 of these contracts and EDGAR's "Source:" lines of 35) |
| | Gold evidence spans found again in the app's text | 813 / 815 |
| | Contracts with headings found | 102 / 102 (68 headings on average) |
| | Time per contract | 0.06 s on average; the slowest took 0.33 s (106 pages), not counting a one-off 1 s start-up on the first file |
| 3 BM25 | Answerable questions with a gold span inside the top 10 chunks | 90.1% |
| 4 DeBERTa | Accepted answers that have gold evidence (precision) | 97.4% |
| | Answerable questions accepted (recall) / with the best span on gold evidence | 83.5% / 81.9% |
| 5 Expand | Accepted answers with a section path / with page numbers | 100% / 100% |

- **Reading the PDF costs nothing measurable.** On CUAD's text the same model
  (the DeBERTa run's own test predictions) gets precision 97.4%, recall 84.0% and
  span hit 82.2%, and 1,004 of the 1,020 accept/abstain decisions are the same.
- **Noise level:** the PDF-reading fix described below changed only the text,
  and it moved span hit from 83.1% to 81.9%.

### Rewrites (steps 6 and 7)

The 380 accepted clauses, next to Run 6's scores on its own validation set
(SimpleGAL, 20 pairs, from `Grpproj/outputs/metrics/simplegal_val/`):

| Measure | On the test PDFs | Run 6 on SimpleGAL validation |
|---|---|---|
| Passed all three checks, so shown in plain English | **53.7%** | |
| Numbers kept, none added | 93.7% | |
| Negation kept (clauses that have one) | 94.2% | 100% |
| NLI coverage ≥ 0.7 / mean NLI coverage | 58.4% / 0.74 | mean 0.79 [0.71, 0.87] |
| Reading grade (FKGL), source → rewrite | 18.9 → 8.3 | 28.7 → 8.7 |
| Length, rewrite / clause in words | 0.83 | 0.85 |
| Time per rewrite (Gemma) / for the checks | 12.1 s (median 9.8 s) / 0.1 s | |

### Why about half of the rewrites are held back

- **NLI coverage, mostly.** It is the only failing check for 138 of the 176 held back.
- **Short clauses fail more often.** Coverage is a mean over the clause's
  sentences, so one loosely paraphrased sentence pulls a two-sentence clause
  under the bar. 44% of clauses under 80 words reach 0.7, against 88% of
  clauses of 150–300 words.
- **The bar is strict even for good rewrites.** Of the 885 synthetic pairs Run 6
  was trained on, 70% reach 0.7.
- **Wrong spans are not the cause.** On the 363 answers whose span is gold
  evidence, 52.6% pass.

### A reading of 40 answers

The first 40 answers were read in full. 24 were shown and 16 held back.

| Held back because | Answers | Examples |
|---|---|---|
| A real error | 3 | A reversed right: "either party may terminate ... so long as no default exists" became "Neither party can end this Agreement ... if no default exists" (NLI coverage 0.00). A deadline moved from after a date to before it. A definition's scope was dropped (caught by the numbers check). |
| A loose paraphrase | 3 | "construed" became "built"; "for an indefinite period" became "forever" |
| A false alarm | 10 | "unless" paraphrased as "if"; NLI coverage just under 0.7; a heading pulled in from the next section by the 40-word expansion; number words ("Five Million Dollars ($5,000,000)" counts a 5 the rewrite does not repeat) |

None of the 10 shown rewrites read in full changed the meaning.

### A fix made during the evaluation

The first full run of this evaluation found page furniture left in the text:

- page numbers and EDGAR's "Source: ..." lines in mid-page, because web pages
  printed to PDF put several original pages on one PDF page
- clause numbers set apart from their titles, such as "11." left at the end of
  the clause before "INTELLECTUAL PROPERTY"

A page number at the end of a clause counted as a number the rewrite had
dropped. The PDF reader was fixed, tests were added, and the evaluation was run
again; the numbers above are from that second run. The fix raised the numbers
check from 86.8% to 93.7% and all checks from 50.6% to 53.7%. No threshold or
model setting changed.

Two gaps remain in the PDF reader. Both are listed under Limitations in the
README.

- A sentence split by a page break is joined again only when the next page
  starts in lower case. 37 of the 1,618 page breaks in the test PDFs stay split
  before a capitalised word.
- A running header that repeats lower than the top 8% of the page stays in the
  text.

```powershell
docker compose run --rm jargonaut-app python scripts/eval_end_to_end.py --gemma-contracts -1   # about 80 minutes
```

## The fictional contract test set

Kiara's `test` branch adds ten invented one-page contracts
(`Fictional_Contract_Test_Set/`), each written around one of the 10 topics.
They come without answers. Their README says they are for demonstration, not a
substitute for held-out CUAD evaluation. The set is small: one of its 28
answerable pairs moves recall by 3.6 points, so counts are given with the rates,
and the brackets are 95% Wilson intervals.

### How it was run

- **Blind:** the PDFs are read as `doc_a.pdf` … `doc_j.pdf`, in shuffled order
  (`eval/fictional/blind_map.json`).
- **Gold labels came before any model ran.** All 100 contract × topic pairs were
  labelled, not just each contract's own topic, following a written guide
  (`eval/fictional/labeling_guide.md`).
  - Under the guide, a sentence that only denies a topic is not evidence. Examples
    are "There is no automatic renewal" and a "nonexclusive license". Such
    sentences are listed as near-misses: the set's traps.
  - Two annotators labelled independently. Both were Claude agents, not people.
    The second saw only the guide and the contracts' text.
  - They made the same call on all 100 pairs (28 with evidence, Cohen's κ = 1.0),
    and their quoted evidence overlaps on all 28.
  - In 2 pairs one quoted more than the other, and `gold.jsonl` keeps both quotes.
  - The label files were hashed before the run (`frozen.sha256`).
  - Kiara wrote the contracts, so she is the right person to check `gold.jsonl`.
- **Settings:** the app's own, unchanged (accept at a score of 6.25).
- **Retrieval is not tested.** Each contract is about 400 words, which makes 2
  chunks, so BM25's top 10 is always the whole contract. The set tests DeBERTa
  and its decision to answer or abstain.

### Extraction: 100 contract × topic pairs

The full CUAD question for each topic, scored as in `deBERTa_training.ipynb`
section 8:

| Measure | Fictional set | CUAD test PDFs (above) |
|---|---|---|
| Precision: accepted answers that are evidence | **100%** (19/19) [83–100%] | 97.4% |
| Recall: answerable pairs accepted | **67.9%** (19/28) [49–82%] | 83.5% |
| Span hit | 67.9% (19/28) | 81.9% |
| Exact match / token F1 (abstentions score 0) | 64.3% / 66.3% | |
| Answerability average precision | 0.995 | |

- **Each contract's own clause: 10 of 10 found**, all on the gold sentence.
- **No false accepts.** None of the 72 pairs without evidence was accepted,
  including the 26 that have a trap sentence. The highest score among them was
  −2.28, far below 6.25.
- **Every miss is an abstention on the right sentence.** For all 28 answerable
  pairs, the best span is the gold sentence. The 9 misses are scores below the
  threshold:
  - **Governing Law: 8 of 10 missed**, with scores of 0.66–4.22. See below.
  - **Expiration Date in the non-compete contract** (score −12.96). Both annotators
    marked this pair as unsure: that agreement has no term, only an end date for
    its restrictions. The model's reading is defensible.
- **By topic:**
  - Governing Law: 2/10.
  - Expiration Date: 8/9.
  - Termination For Convenience: 2/2. One of the two is an incidental clause in
    the governing-law contract.
  - Every other topic: 1/1.
  - The Exclusivity answer covers only the first of the two gold sentences
    (token F1 0.57).
- **Incidental clauses are where recall drops.** Topics a contract mentions in
  passing are found 9 times out of 18. All 9 misses are the abstentions listed
  above.

### Why Governing Law is missed: the wording

`scripts/probe_governing_law.py` replaces only the governing-law sentence of each
contract and scores the Governing Law question again. Read as text, the
contracts as written give the same scores as their PDFs.

| Governing-law sentence | Accepted | Scores |
|---|---|---|
| As written, e.g. "New York law governs." | 2 / 10 | 0.66–10.50 |
| "This Agreement is governed by New York law." | **10 / 10** | 6.42–8.42 |
| "The laws of the State of New York govern this Agreement." | 3 / 10 | 4.78–7.03 |
| "This Agreement shall be governed by the laws of the State of New York." | **10 / 10** | 8.34–9.69 |
| CUAD-style: "… shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of laws principles." | **10 / 10** | 9.42–11.22 |

- **Location is not the problem.** In all 50 versions the best span is the
  governing-law sentence.
- **The verb form decides.** The passive "governed by" is accepted every time.
  The active "… law(s) … govern(s)" mostly is not, even with "the laws of the
  State of".
  - The two accepted as written are "shall be governed by New York law" and
    "Delaware law governs this Agreement, subject to applicable mandatory law",
    an active sentence with a longer tail.
- **It mirrors the training data.** Of CUAD's 464 governing-law evidence spans,
  364 (78%) say "governed by" or "construed in accordance with/under". Only 5
  (about 1%) use the active form these contracts use.
- **What it means:**
  - On real filings the boilerplate dominates, and Governing Law recall is 0.95
    on the CUAD test PDFs.
  - Contracts drafted in plain style would lose it.
  - Nothing was changed here. Lowering the threshold on this set would be tuning
    on the test. Training with reworded contexts is the clean fix.

### Questions through the whole app

The second annotator also wrote 60 plain-English questions (`questions.jsonl`):

- For each contract, 4 about topics the contract covers and 2 out of scope: one
  about something else in the contract (fees, warranty, confidentiality), one
  unrelated.
- Each question goes through the router, extraction, the Gemma rewrite and the
  checks, as in the app.
- When the router asks which topic was meant, the right one is picked, as a user
  would.

| Measure | Result |
|---|---|
| In-scope questions routed to the right topic | 39 / 40 (1 after a clarification) |
| Out-of-scope questions rejected | 19 / 20 |
| In-scope questions shown the right evidence | 31 / 40: 8 governing-law abstentions, 1 routing miss |
| Rewrites that pass all three checks (distinct clauses) | 16 / 19 |
| Time per question | mean 4.9 s, longest 17.2 s |

- **Routing errors:**
  - "once I pay, do I actually own the artwork?" went to out of scope
    (probability ≈ 1.0), though it is about IP ownership.
  - "how long are the trays guaranteed for?" asks about the warranty, but went to
    Expiration Date (0.75). The user would get the contract's end date as the
    answer.
  - "is the supplier allowed to go around me and sell straight to customers in my
    area?" led to a question back: exclusivity (0.65) or non-compete (0.35)?
    Choosing exclusivity gave the right clause.
- **The 3 held-back rewrites:**
  - **Insurance and Cap On Liability:** NLI coverage of 0.671 and 0.669, just under
    0.7. Both rewrites keep every amount and condition. The cap rewrite drops
    "Subject to Section 4", which points to the cap's exceptions.
  - **Non-Compete:** NLI coverage 0.96, but it failed the numbers check.
    "eighteen months" became "18 months", and the number-word map in `checks.py`
    has no "eighteen". This is a false alarm. The map is Grpproj's code, kept as
    it is for parity.
- **Rewrites pass more often here than on CUAD (16/19 against 53.7%).** These
  clauses are short and cleanly drafted, and none has PDF furniture or a heading
  pulled in from the next section.

### Limits of this set

- It has ten contracts by one author, with one target clause each: a sanity
  check, not a benchmark.
- Both annotators and the question writer were language-model agents. Their
  labels agree with each other, but no person has checked them yet.

```powershell
docker compose run --rm jargonaut-app python scripts/eval_fictional.py            # about 8 minutes with model loading; --no-gemma for extraction only
docker compose run --rm jargonaut-app python scripts/probe_governing_law.py       # a minute or two (DeBERTa only)
```

## Where the numbers come from

- **Saved results:**
  - `eval/router_results_{gemma,fallback}_{dev,test}.json`
  - `eval/calibration.json`
  - `eval/end_to_end/`: `summary.json`, `per_question.jsonl`, `documents.jsonl`, and `examples.json` with the first 40 answers in full
  - `eval/fictional/`:
    - inputs: the blinded PDFs, `blind_map.json`, `labeling_guide.md`, `gold_a.jsonl`, `gold_b.jsonl`, `gold.jsonl`, `questions.jsonl` and `frozen.sha256`
    - results: `summary.json`, `per_pair.jsonl`, `per_question.jsonl`, `documents.jsonl`, `probe_governing_law.json` and `run_log.txt`
- **Parity table:** what `scripts/check_parity.py` prints. It saves no file; rerun it to regenerate the table.
- **"Methods tried on dev":** these figures come from development runs whose outputs were not kept.
- **The 40-answer table:** sorting the answers into real errors, loose paraphrases and false alarms is a manual reading of `examples.json`.
- **Fictional set:**
  - The judgements on the 3 held-back rewrites come from reading them in `per_question.jsonl`.
  - The CUAD wording counts come from the Governing Law answers in `qa.jsonl.gz` (all splits), sorted by pattern: "governed by" or "construed in accordance with/under", against an active "law(s) … govern(s)".
- **Run 6's scores:** `Grpproj/outputs/metrics/simplegal_val/summary.md`.
- **Coverage of the synthetic pairs:** `Grpproj/TRAINING_PROGRESS.md`.
