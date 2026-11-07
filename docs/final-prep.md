# Final preparation plan

Prepared 7 November 2026 from the current project tree, proposal, course guidelines, latest report and presentation drafts, saved evaluation outputs, and source code. Use this as the top-level checklist; the report and presentation edit briefs are [final-report.md](report/final-report.md) and [final-presentation.md](presentation/final-presentation.md). These briefs track issues and recommended edits; the PDF files must still be checked after each update.

## Current project map

| Location | Contents |
| --- | --- |
| `deberta-docker/` | DeBERTa training notebook and Docker setup |
| `gemma-eval-docker/` | Gemma evaluation notebook and Docker setup |
| `jargonaut-app/` | Application, tests, evaluation scripts, and saved results |
| `docs/proposal/` | Project proposal PDF |
| `docs/guidelines/` | Course project guidelines PDF |
| `docs/report/` | Latest report draft PDF, extracted text, and `final-report.md` edit brief |
| `docs/presentation/` | Presentation draft PDF and `final-presentation.md` edit brief |
| `docs/evaluation/` | Evaluation results DOCX and `REVIEW_FINDINGS.md` |
| `docs/test-data/` | Ten fictional test contract PDFs and README |
| `docs/final-prep.md` | This checklist |

The supplied folder currently contains the individual fictional contract PDFs rather than the original ZIP. The application’s blinded evaluation copies remain in `jargonaut-app/eval/fictional/pdfs/`. The reviewed set had matching extracted text across the original and blinded versions.

## Priorities before submission

### 1. Resolve provenance and explain deviations from the proposal

- Document how the 886 Gemma training pairs, 220 router questions, and fictional labels were produced; name the actual tools and interfaces used, including whether an API was called.
- The proposal planned Claude/OpenAI-generated questions, while the course guideline excludes LLM APIs. Establish the actual workflow and seek instructor guidance on its acceptability. Do not describe agent-labelled examples as human annotation.
- Clarify whether ContractNLI contributed to Gemma’s training data. DeBERTa’s `syn0` run setting only indicates that synthetic extractor questions were disabled.
- Explain the move from planned Gemma 4 31B QLoRA to Gemma 4 E4B LoRA using verified training records and hardware details.
- State that the RoBERTa baseline was not run if that is confirmed. Do not imply the proposed baseline or ablations were completed.
- Confirm the actual contributions of all team members rather than copying the proposal’s role split unchanged.

### 2. Make the results precise and comparable

- On CUAD test PDFs, use **answerability precision** for 370/380 = 97.4%. Separately report **accepted-span overlap precision** 363/380 = 95.5%, **answerability recall** 370/443 = 83.5%, and **span-hit recall** 363/443 = 81.9%.
- Label FKGL 8.3 as the average of generated rewrites before the check gate. When failed rewrites are replaced by original clauses, the approximate mean of the displayed text is 13.33; give the averaging method and do not mix it with the separate 20-clause validation score.
- For the fictional extraction set, include denominators: 19/19 accepted evidence-bearing pairs, 19/28 recall, 0/72 false accepts, and 10/10 target clauses hit. Explain that the short contracts produce two chunks each, so this is not a demanding retrieval test.
- Separate the router benchmark (110 questions) from the fictional full-app questions (60). The 39/40 in-scope routing count includes one clarification resolved using the expected topic; direct routing was 38/40.
- Keep the 16/19 distinct fictional rewrites passing checks separate from the 24/31 rewrite events in the saved full-app summary.
- Report SimpleGAL validation (20 clauses), CUAD test PDFs, router test, and fictional demonstration as separate evaluations with their own sample sizes.

### 3. Bound the claims about rewrite safety

- Describe the current gate as heuristics: number-set equality, presence of a negation cue, and backward NLI coverage of at least 0.7. It calculates forward support and contradiction scores but does not use them in pass/fail.
- Say the gate defaulted to the original clause for the documented reversed termination example. Do not claim it prevents all meaning reversals. The interface permits users to inspect a rewrite that failed the checks.
- Mention known limits where relevant: “eighteen” may fail against “18”; quantities are not tied to units or parties; a negation moved from one party to another can pass the presence check.
- The cap example drops “Subject to Section 4,” a reference to exceptions. Do not describe it as preserving every condition.
- Explain that user wording is routed to a topic and replaced by a fixed CUAD question for retrieval and extraction. The system may not answer all specifics in a natural-language question. Referenced definitions are not retrieved.

### 4. Finish, render, and verify the deliverables

- Work through [the report edit brief](report/final-report.md). The latest report PDF is 11 pages; the course limit is 15 A4 pages, minimum 11 pt, including references, figures, and appendices. Check page count after further revisions.
- Work through [the slide edit brief](presentation/final-presentation.md). Remove all placeholders, add confirmed names/date and references, and rehearse within the 25-minute limit. Prepare a demo fallback.
- Complete a reproduction guide with environment versions, GPU needs, exact commands, external model/data locations, and asset versions or checksums. The full app currently relies on external model weights, caches, and datasets not present in this tree; Docker is configured for offline model loading.
- Update the root `README.md` if its folder map or setup instructions no longer match the present project structure.
- Render the final report and slides after editing. Inspect every page for clipped text, split numbers, broken tables, missing references, and remaining TODOs.

## Interpretation guardrails

- BM25 is retrieval, DeBERTa is the answer-span extractor, Gemma E4B is used for simplification and routing, and the NLI model is part of the checker. The proposal’s RoBERTa baseline has no result in the supplied files.
- The app contains a lightweight dialogue feature for clarification, active-topic follow-up, and “anything else?” requests. Describe those implemented behaviors without implying unrestricted conversational understanding.
- Saved outputs were checked for internal consistency, but full model inference was not rerun because required external assets and dependencies are absent from this checkout. Avoid presenting the saved results as independently reproduced in a clean environment.

## Latest report draft: remaining checks

The newest report PDF incorporates several earlier corrections: it now distinguishes direct routing from routing after clarification, describes the cap example as omitting cross-references, labels 8.3 as before gating, acknowledges heuristic limits, states DeBERTa used no synthetic QA or ContractNLI, and says the RoBERTa baseline was not run. The PDF is 11 pages.

Still verify or correct these items before treating the report as final:

- The abstract still calls 97.4% “precision against gold spans.” It is **answerability precision** (370/380). Add accepted-span overlap precision separately: 363/380 = 95.5%; answerability recall is 370/443 = 83.5%; span-hit recall is 363/443 = 81.9%.
- Page 2 says the fictional labels were manually labelled and team-reviewed. Confirm who did this and how; earlier evaluation notes identify Claude agents as annotators. Ensure the description accurately distinguishes agent-generated labels from human review.
- Page 4 contains revised DeBERTa settings and says the batch size was “being confirmed against training logs.” Resolve that note against the actual run records; notebook settings are train batch 8 with accumulation 4 (effective 32), inference batch 16, and learning rate 1e-5.
- Page 4 still needs reproducible commands. Verify every command against the actual repository and document the external data/model/cache mounts; do not leave placeholder `src/...` or Streamlit commands.
- Page 7’s manual-review count/process remains to be confirmed. Remove the assertion of full manual review if the reviewer/sample cannot be substantiated.
- Page 10 says all references were verified. Confirm that review actually happened; remove the duplicate Platt entry if it remains and add the actual Gemma 4 E4B and NLI model references.
- Check whether the presentation has been updated to match the new report and whether all team contributions are confirmed.

Current source and model folders remain in place; project reference documents are grouped under `docs/`. The report and presentation Markdown files are edit briefs, not revised exports. The newest report PDF has changed; the presentation PDF remains to be checked against its brief.
