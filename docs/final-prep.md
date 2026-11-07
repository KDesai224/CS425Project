# Final preparation plan

Prepared 7 November 2026 from the proposal, course guidelines, current report and presentation PDFs, saved evaluation outputs, and source code. This is an action list, not a claim that outstanding work has already been completed.

## What is in the project

Jargonaut combines BM25 retrieval, a CUAD fine-tuned DeBERTa extractor, a Gemma 4 E4B LoRA simplifier/router, and an NLI model with lexical checks. There are saved CUAD PDF evaluation results for 102 contracts and a separate synthetic set of ten fictional contracts. The RoBERTa baseline promised in the proposal is not present. The current checkout does not contain the trained model files, Gemma training code, several shared datasets, or all external model caches required by Docker. The existing results can be checked for internal consistency, but cannot be fully reproduced from this folder alone.

## Priority order for the remaining preparation

### 1. Resolve provenance and proposal deviations

- Record the actual origins of the 886 Gemma training pairs, the 220 router questions, the fictional set labels, and any generated augmentation. State the tools and interfaces used, including whether an API was called.
- The course guideline says not to use LLM APIs. The proposal itself proposed offline Claude/OpenAI generation. Ask the instructor how that applies to the actual workflow and accurately report the answer. Do not call agent-labelled examples human annotated.
- State whether ContractNLI contributed to Gemma's synthetic training pairs. The DeBERTa run has `syn0`; that fact alone does not show ContractNLI was absent from the separate Gemma training pipeline.
- Explain why Gemma E4B with LoRA replaced the planned 31B QLoRA configuration. Verify hardware, training settings and selected checkpoint from run logs/configs before stating them.
- State clearly that no RoBERTa baseline was run if no valid run exists. Do not imply the proposed ablation was completed.

### 2. Correct the quantitative story

- Use **answerability precision** for 370/380 = 97.4%: accepted questions whose contract has gold evidence.
- Separately report **accepted-span overlap precision**: 363/380 = 95.5%, and **span-hit recall**: 363/443 = 81.9%.
- Call 8.3 FKGL the mean of generated rewrites before the check gate. Since 176 of 380 rewrites are withheld, the output shown to users has an approximate per-answer mean FKGL of 13.33 when the original clause is shown on failure. State the averaging method and keep it distinct from the 20-clause controlled validation set.
- For the fictional set, report the denominators: 19/19 accepted evidence-bearing pairs, 19/28 recall, 0/72 false accepts, and 10/10 target clauses hit. Emphasize that each document is about 400 words/two chunks and top-10 retrieval therefore does not meaningfully test difficult retrieval.
- Routing: 39/40 in-scope are correct when one clarification was resolved with the expected user choice; 38/40 were directly routed correctly. The 60-question full app run is distinct from the 110-question router benchmark.
- Distinguish the 16/19 distinct fictional rewrite clauses passing checks from 24/31 question events in the saved end-to-end summary.
- Keep validation (20 SimpleGAL clauses), CUAD test PDFs, and fictional demonstration results in separate tables and prose.

### 3. Make the safety and quality claims precise

- The rewrite gate checks number-set equality, presence of a negation cue, and backward NLI coverage >= 0.7. It computes forward support and contradiction but does not gate on them. A passing label is a heuristic outcome, not legal equivalence certification.
- The application defaults to the original clause for failed checks but exposes the failed rewrite behind a disclosure. Say that the documented reversal was caught; don't promise all meaning reversals are blocked.
- State the observed lexical limitations: number words such as “eighteen” can fail against digits; units/roles are not bound to numbers; a negation shifted between parties can pass the presence test.
- Mention the missed cross-reference “Subject to Section 4” in the cap example. It points to exceptions and should not be described as preserving every condition.
- Clarify that the user question is routed to a topic and then replaced by the fixed CUAD category question for retrieval/extraction. Specific question constraints are not necessarily answered. Definitions referenced by the clause are not fetched.

### 4. Finish and verify the deliverables

- Follow `final-report.md` in order. The guideline limit is 15 A4 pages, minimum 11 pt including figures/references/appendices; the current PDF is 12 pages, so additions need page-budget control.
- Follow `final-presentation.md` for slide-by-slide work. The guideline allows up to 25 minutes and asks that 90% of results be ready. Remove every placeholder and rehearse the demo, questions, and a backup recording/screenshots.
- Replace inherited proposal task assignments with actual contributions, confirmed by the team.
- Supply complete references for CUAD, DeBERTa, BM25, LoRA, Gemma model card, NLI model card, SimpleGAL/SimpLegal-Pref source, SARI and readability measure. Use only sources actually used.
- Add a reproduction guide listing Python/container versions, GPU needs, exact commands, external folders and asset versions/checksums. The current project root README is stale and describes an empty proposal repository; update it to point to the organized `docs/` tree and actual code.
- Before submission, visually inspect every rendered page/slide and confirm there are no split decimals, broken tables, placeholders, clipping, or unreadable references.

## Folder organization performed

- `docs/proposal/`: proposal PDF
- `docs/guidelines/`: course guideline PDF
- `docs/report/`: original report draft PDF
- `docs/presentation/`: original presentation draft PDF
- `docs/evaluation/`: evaluation DOCX and review findings
- `docs/test-data/`: fictional contract test ZIP
- `deberta-docker/`, `gemma-eval-docker/`, `jargonaut-app/`: code and saved project evaluations retained in place
- `tmp/archive/review/`: text extractions, rendered page images, and notebook inspections created during review

Original drafts and data were preserved. No trained model or code directory was moved.
