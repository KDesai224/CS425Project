# Jargonaut app: the whole system

Upload a contract (PDF, DOCX or TXT), ask a question in your own words, and get
a plain-English answer. Each answer comes with the original clause, its section
and page, and a confidence score. The app joins the team's components
(Theo's data prep, Kiara and Ridheema's BM25 and DeBERTa, Ian and Shernice's
Gemma adapter) into the system described in the proposal. No model was retrained. What was
measured is in [FINDINGS.md](FINDINGS.md).

```text
upload contract (PDF/DOCX/TXT)
   │  1. Ingest: text with page numbers and section headings kept,
   │     chunked like Theo's data (300 words, 60 overlap)
   ▼
user question ──► 2. Route to one of the 10 CUAD categories (or "out of scope")
   │
   ▼  3. BM25: top 10 chunks for that category's CUAD question
   ▼  4. DeBERTa: best answer span + confidence score
   │        below the threshold ──► "Not enough evidence in this contract." (Gemma never runs)
   ▼  5. Expand the span to its whole clause (40-300 words)
   ▼  6. Gemma 4 E4B + Run 6 adapter: plain-English rewrite (training system prompt)
   ▼  7. Checks: numbers kept? negations kept? NLI coverage?
   ▼
answer = rewrite + original clause + section/page + confidence
         (checks failed: the original clause with a warning)
```

## Run it

Like `deberta-docker/` and `gemma-eval-docker/`, this needs Docker Desktop with
GPU support, and it mounts models that are already on this machine (see
`docker-compose.yml`):

| Mount | From |
|---|---|
| DeBERTa model and `threshold.json` | `../deberta-docker/runs/deberta-v3-large-squad2_all0_syn0_neg3/` |
| Gemma adapter (Run 6) | `../outputs/e4b-r16-lr0.0002-e2-s42-syn/best/` |
| Gemma 4 E4B and the NLI model | `../../Grpproj/.cache/huggingface/` (offline, nothing is downloaded) |

From this folder:

```powershell
docker compose build
docker compose up          # open http://localhost:8765 once the log says "Models ready" (about 2 minutes)
```

The page opens at once. The status pill turns green when the models have loaded.
With all three models loaded the GPU holds about 14 GB.
The port is 8765 because another local web server already uses 8000 on this
machine; set `APP_PORT` to change it.

From the command line, copy the contract into this folder first. The container
sees this folder as its working directory.

```powershell
docker compose run --rm jargonaut-app python ask.py contract.pdf "Can I cancel this early?"
docker compose run --rm jargonaut-app python ask.py contract.pdf --topic Insurance --json
docker compose run --rm jargonaut-app python ask.py contract.pdf          # interactive conversation
```

`LOAD_GEMMA=0` starts without Gemma. Answers then show the original clause
only, and keyword rules with NLI route the questions.

## How each step is done

| Step | What happens | Where it comes from |
|---|---|---|
| 1 Ingest | Text is rebuilt one paragraph per line, as in CUAD's text. Wrapped lines are rejoined, and running headers, footers and page numbers are removed, also where web pages printed to PDF (most EDGAR filings) leave them in mid-page; a sentence they cut is joined again. Page breaks and headings are kept as character offsets, so any span maps back to its page and section path. | New: `jargonaut/ingest.py` |
| 1 Chunk | Windows of 300 words with a 240-word stride | Theo's `make_chunks.py`, unchanged |
| 2 Route | Gemma, with its adapter switched off, picks one of 11 options (the 10 topics or "none of these") in a single forward pass. If the runner-up topic has at least 25% probability, the app asks which one was meant. | New: `jargonaut/router.py` |
| 3 BM25 | Query form B (category name + description), NLTK stopwords minus no/nor/not/without, Porter stemming, k1=1.5, b=0.75, one index per contract | `BM25_Retrieval_and_Evaluation.ipynb`, final setup |
| 4 DeBERTa | Full CUAD question with each chunk in 512-token windows (stride 128), top 20 start/end logits, spans up to 256 tokens. Score = best span − null score; accepted at ≥ 6.25. | `deBERTa_training.ipynb` `predict()` and the local run's `threshold.json` |
| 5 Expand | Whole paragraphs until 40 words, then cut back to 300 words around the span | `Grpproj/scripts/02_build_pipeline_set.py`, which also built Run 6's training sources |
| 6 Rewrite | `SYSTEM_PROMPT`, chat template with thinking off, 4-bit NF4, greedy, ≤ 512 new tokens | `Grpproj/jargonaut/common.py` |
| 7 Check | Every number in the clause is kept and none is added; a negation is kept if the clause has one; NLI coverage ≥ 0.7 | `Grpproj/jargonaut/metrics.py`. 0.7 is the bar Grpproj step 10 used for the "synthetic strict" set. |

The proposal also asks for "a lightweight dialogue" that "tracks the active
contract/topic for follow ups and requests clarification when needed".
`Conversation` in `jargonaut/pipeline.py` does this:

- It asks which topic was meant when two are close; a reply of "1", "2" or a topic name resolves it.
- "Anything else?" shows the next-best accepted passage for the active topic.
- It lists the topics on "help".

## Tests and findings

```powershell
docker compose run --rm jargonaut-app python -m pytest -q tests          # 35 unit tests, no GPU
```

What was measured, how, and how to rerun it is in [FINDINGS.md](FINDINGS.md). In short:

- The reused components give the team's saved outputs back exactly (6 of 6 parity checks).
- 97.3% of held-out test questions are routed to the right topic, and none in scope goes to a wrong one.
- Reading CUAD's PDFs costs nothing measurable. The best span is on gold evidence for 81.9% of answerable test questions, against 82.2% on CUAD's own text.
- 53.7% of rewrites pass all three checks and are shown in plain English (reading grade 18.9 → 8.3). The rest show the original clause with a warning. In a reading of 40 answers, most of those were false alarms.

## API

The page uses this JSON API, which another frontend can call too:

| Request | Returns |
|---|---|
| `GET /api/health` | `{"ready", "status", "error", "gemma", "calibrated"}` |
| `GET /api/topics` | the 10 categories with their display names |
| `POST /api/documents` (multipart `file`) | `doc_id`, pages, words, chunks, headings, warnings, outline |
| `POST /api/documents/{doc_id}/ask` with `{"question": "..."}` or `{"category": "Insurance"}` | an answer (below) |
| `DELETE /api/documents/{doc_id}` | forgets the contract |

An answer has a `kind`, which is one of:

- `answer`: the checks passed
- `unchecked`: the checks failed, so the original clause is shown with a warning
- `no_evidence`
- `out_of_scope`
- `clarify`
- `help`

The rest of the answer is: `message`, `topic`, `rewrite`, `clause`,
`highlight` (the span's position in the clause), `section` (heading path),
`pages`, `confidence` (`score`, `threshold`, `probability`), `checks`,
`retrieved` (the BM25 top 10), `route` and `timings`.

## Files

| Path | Contents |
|---|---|
| `jargonaut/` | the pipeline: `ingest`, `chunking`, `router`, `retrieval`, `extractor`, `expand`, `simplifier`, `checks`, `pipeline`, `config` |
| `app.py`, `static/index.html` | web app and UI (no external resources) |
| `ask.py` | command line |
| `scripts/` | `check_parity.py`, `calibrate.py`, `eval_router.py`, `eval_end_to_end.py`, `eval_fictional.py` |
| `tests/` | unit tests: ingestion of generated PDF/DOCX/TXT files, chunking, expansion, BM25, rules, checks |
| `eval/` | the router question set and results, `calibration.json` (read by the app), end-to-end results, and `fictional/` (the `test` branch's contracts, blinded, with gold labels, questions and results) |
| `FINDINGS.md` | what was measured: parity, routing, confidence, end to end |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `APP_PORT` | 8765 | host port of the web app |
| `LOAD_GEMMA` | 1 | 0 = no Gemma: clauses only, rule + NLI routing |
| `NLI_COVERAGE_MIN` | 0.7 | NLI coverage needed to show a rewrite |
| `DEBERTA_RUN_DIR`, `ADAPTER`, `GRPPROJ` | sibling folders | where the models are |
| `DATA_PREP`, `EVIDENCE`, `CUAD_PDF` | sibling folders | data for `scripts/` only |

## Limitations

- **Ten topics only.** Any other question is out of scope, by design.
- **One clause per answer.** Definitions the clause refers to are not pulled in (the proposal mentions this). "Anything else?" shows further passages one at a time.
- **The checks are strict heuristics.** On the test PDFs they hold back 46% of the rewrites. In a sample of 16 held back, 10 were false alarms (see [FINDINGS.md](FINDINGS.md)).
  - `negation_kept` counts "unless" and "except" as negations. So "unless extended in writing" → "it can run longer if they agree in writing" fails, though the meaning is kept.
  - NLI coverage is noisy on short clauses.
  - Both are the team's own metrics, kept as they are so the app and the Gemma evaluation agree.
  - A false alarm shows the original clause with a warning, so the cost is a missed simplification, not a wrong one.
- **Clause expansion is the team's, unchanged.** To reach 40 words it adds neighbouring paragraphs, which can be the next section's heading or first sentence. The rewrite then drops the heading, and NLI coverage counts it as missing.
- **Gemma is slow.** A rewrite takes 12 s on average (median 9.8 s; 1 in 10 takes over 23 s) in 4-bit with the adapter unmerged, the setting the adapter was validated in.
- **PDF limits.**
  - Scanned PDFs (no OCR) and legacy `.doc` files are refused with a message.
  - Multi-column layouts are read in the order the PDF stores them.
  - A sentence split by a page break is joined again only when the next page starts in lower case. 37 of the 1,618 page breaks in the test PDFs stay split before a capitalised word, such as a defined term.
  - A running header that repeats lower than the top 8% of the page stays in the text, for example the confidentiality legends of some EDGAR exhibits.
- **Word files need a saved layout for page numbers.** Pages are only shown when Word saved its page layout (`w:lastRenderedPageBreak`), which Word does on save. Files generated by other tools get sections but no page numbers, and the app says so.
- **The router set was written by a model.** It is independent of the router but not of language models in general; real users may phrase things differently.
- **Not legal advice.** The UI says so.
