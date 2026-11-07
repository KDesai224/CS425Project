# Report revision instructions

Use this as the detailed edit brief for the report draft. Keep the final report within **15 A4 pages, 11 pt minimum**, counting figures, references and appendices as required by the course guideline. The latest PDF reviewed is 11 pages; render the revised document before submission. The editable source file was not present in the project at review time, so these are specific edits to apply to the source document rather than direct edits to the PDF.

## Page 1: title, abstract and contribution

- Replace “DRAFT”, date-specific draft label, TODOs, and author-name formatting inconsistencies. Confirm all five names/spellings and group ID with the team.
- Rewrite the abstract to distinguish: 97.4% answerability precision; 95.5% accepted-span overlap precision (363/380); 83.5% answerability recall; 81.9% span-hit recall (363/443). Explain that the 8.3 FKGL is the mean for all generated rewrites before gating, while the original is shown for failed checks; the approximate mean of the actually displayed output is 13.33.
- Replace “checks keep reversed meanings from reaching the user” with a bounded claim: the checks caught a specific reversed-termination example and fall back to the original by default when a heuristic fails. Note the failed rewrite remains inspectable and the checks do not certify legal equivalence.
- Complete the contribution paragraph with actual changes relative to the proposal, not a list of components alone: full upload-to-answer integration, traceable pages/headings, topic router, clarification, abstention calibration, and rewrite gate. Confirm what was new versus reused from team components.

## Pages 2–3: proposal changes, data and annotation

- Replace every dataset TODO with a concise provenance table. For each source give actual name/version, purpose, sample size/split, whether used for training/validation/test, and generation/annotation process.
- Reconcile **SimpLegal-Pref** (proposal) versus **SimpleGAL** (current evaluation files). Verify the exact dataset identity and citation before choosing one label; do not guess based on similar names.
- Explain Gemma deviation: proposed Gemma 4 31B with QLoRA, with E4B/LoRA fallback; actual app uses Gemma 4 E4B plus Run 6 LoRA. Give the actual resource constraint only if supported by team records.
- State that Run 6 training used 886 synthetic pairs if verified. Explain actual source, filtering, generator/interface, human checks, split protection, and whether ContractNLI supplied any training examples. The saved “syn0” label applies to DeBERTa synthetic questions, not necessarily Gemma training.
- DeBERTa notebook settings show `USE_SYNTH=False`, `NEG_K=3`, seed 42 and 10 categories. State that the proposal’s plain-language extractor augmentation was not part of this run. Identify the absent RoBERTa baseline and say it was not run, if confirmed.
- The proposal says generated QA uses Claude/OpenAI, while the course guideline excludes LLM APIs. Disclose exactly what was used for all generated QA, synthetic pairs, router questions and labels. Explain instructor guidance if received. Do not describe the two Claude agents in the fictional exercise as human annotators. Kappa=1 reflects agreement, not independent human validity.
- Preserve the fictional-set caveat: ten invented contracts, labels frozen before model evaluation as recorded, demonstration only. The ZIP README warns that this is not a substitute for held-out CUAD. Note that the code’s manifest hash is line-ending sensitive for some files; do not present it as proof of label chronology.

## Page 3: preprocessing and architecture

- Keep the verified pipeline: ingest PDF/DOCX/TXT; 300-word chunks with 60-word overlap; BM25 top 10; route among ten CUAD topics/out of scope; DeBERTa span and 6.25 threshold; expand context; Gemma rewrite; checks; return answer with clause, heading/page and confidence.
- Distinguish CUAD evaluation from natural user questions. CUAD end-to-end evaluation supplies the category directly and bypasses router. In the app, user wording selects a topic, then the fixed CUAD question is sent to retrieval/extraction.
- Describe lightweight dialogue accurately: active topic, clarification when two router scores are close, and “anything else?” retrieving another accepted passage. Do not imply it tracks arbitrary semantic details across follow-ups.
- State that clause expansion may add neighboring paragraphs but definitions referenced elsewhere are not fetched. This only partly meets the proposal’s goal to retain referenced definitions.
- Describe the confidence calibration as a probability that the best span overlaps gold evidence, fitted on validation and checked on test. It is not the probability that the entire answer is legally correct.

## Page 4: model settings and reproduction

- Fill verified extractor details from the notebook: base `deepset/deberta-v3-large-squad2`; 3 epochs; learning rate 1e-5; max sequence 512; stride 128; train batch 8 with accumulation 4 (effective 32); inference batch 16; seed 42; weight decay 0.01; 10% warmup; 3 hard negatives; synthetic question augmentation disabled. Confirm the actual checkpoint/config files before claiming the deployed weights match each setting.
- Include Gemma settings only after checking the Run 6 training config/log: rank 16 and learning rate 2e-4 are in the run name; the selected checkpoint is epoch 1 from an e2 run. Add actual alpha/dropout/target modules/max length/batch/hardware when verified. Do not infer missing values.
- Give exact setup and rerun commands for preprocessing, training, model export, application, evaluation and output paths. Name all required external mounts. This checkout does not contain trained weights, shared CUAD/BM25 data, Gemma training code/sets, or the required Hugging Face cache; explain the reproduction dependency and version/checksum sources. The existing Docker setup is offline mode and cannot fetch missing assets.
- Keep the root README's project map and reproduction note aligned with the report and actual tree.

## Pages 5–8: evaluation results and metrics

- In the metric definitions, define answerability precision/recall separately from span overlap. Recompute the text/PDF table headings so readers know what each denominator represents. The saved CUAD PDF data support: 1,020 questions, 443 with evidence, 380 accepted; 370 accepted on evidence-bearing questions; 363 accepted spans overlap gold. Therefore answerability precision=370/380=97.4%; accepted-span overlap precision=363/380=95.5%; answerability recall=370/443=83.5%; span-hit recall=363/443=81.9%.
- On CUAD text, distinguish model extraction values from PDF ingestion path. Preserve 84.0% recall, 82.2% span hit; 1,004/1,020 accept/abstain decisions agree between paths. Avoid implying PDF parsing has no effect at all.
- Correct “Governing Law accounts for 8 of 10 misses”: there are 9 total answerability misses, 8 in Governing Law. Say “8 of 9 misses” or “8 of 10 Governing Law examples.”
- Explain per-category counts and small samples. The report says the four weakest categories each have fewer than 35 positive examples, but ensure category selection is based on their recall ordering and state the counts.
- Calibration: distinguish ECE/Brier on the full test set from accepted-answer confidence. Define “High” threshold and mention the 0.4–0.8 bins are overconfident. Do not imply calibration measures legal correctness.
- Router: 110-question benchmark is separate from the 60-question fictional full-app test. List benchmark split and error cases. Fictional 39/40 includes one resolved clarification using the expected topic; direct routing is 38/40.
- Simplification: Run 6 SimpleGAL 20-clause results are validation only; there is no Run 6 result on the held-out 40-clause SimpleGAL test. Keep earlier runs’ test results labelled separately and say whether they informed later choices. Include number recall (0.75) alongside number precision (1.00): precision does not show how many source quantities survived. Keep confidence intervals and paired comparison cautious due to n=20.
- Full-system rewrite FKGL: 8.3 is the average generated rewrite across all 380 accepted extractions, including 176 rewrites withheld. State the approximate displayed-output per-answer average (~13.33, original clause used for failed checks) if space and method are confirmed. Do not conflate with controlled validation FKGL=8.74 for Run 6.
- The gate passes 204/380=53.7%; numbers check passes 93.7%; negation check passes 95.8% overall and 94.2% among clauses where negation applies; NLI threshold passes 58.4%. Make denominators explicit. 138/176 held backs fail only NLI.
- Replace claim that the checks “keep reversed meanings from reaching the user.” They compute but do not enforce forward support or contradiction, and the failed rewrite can be inspected in the UI. Present the observed reversal as a useful case, not a safety guarantee.
- The reading of 40 answers needs one reconciled count: text says 24 shown/16 held back but also says only 10 shown rewrites were read in full. State how many items were actually read, selected, and by whom. Another section says no human rewrite review has occurred. Resolve those statements; if only a subset was reviewed, describe it as a subset.
- Do not call the “Subject to Section 4” example a complete condition-preservation success: the report says that exception cross-reference was dropped. State the omission and its implication.

## Pages 9–15: samples, analysis, limitations and references

- Keep examples, but ensure the clause, rewrite and system outcome match saved outputs. State when a rejected rewrite is shown only as an example and that the product defaults to the original clause.
- Explain why CUAD-style boilerplate works better than active plain wording using the documented governing-law diagnostic; frame it as a small synthetic diagnostic, not broad proof.
- Include limitations: 10-topic scope; category routing followed by fixed query; one evidence passage; no referenced-definition retrieval; strict and imperfect heuristics; no OCR; incomplete reproduction assets; no RoBERTa baseline; small controlled rewrite validation; fictional set is synthetic; annotation provenance; and no legal advice.
- Replace future work with realistic next steps: clean reproducibility package, human review with a defined rubric, evaluate on real user questions, study synthetic question augmentation under split discipline, test broader paraphrase/robustness, improve quantity/party/exception checks, retrieve definitions, and run the missing baseline if time/data allow. Separate completed work from future work.
- Update team contribution table from confirmed actual contributions.
- Verify all references and add authoritative model cards/dataset sources actually used. Remove ContractNLI reference only if unused; retain it if it supplied Gemma training material or another component. Add Gemma 4 and NLI model references. Ensure citations are complete and consistently formatted.

## Layout edits observed

- The latest report PDF is 11 pages and appears to meet the page/11-pt requirements. Keep within 15 pages after edits.
- Page 7 table 7 wraps decimals into separate lines (e.g. 37.19 becomes 37.1 and 9). Widen the metric columns or reduce wrapping without reducing text below 11 pt.
- Table 8 caption is stranded at the bottom of page 7, with its body on page 8. Keep caption and table together.
- Check the reliability table headers on page 6 and all references after content changes.

## Corrections identified in the newest report draft

Apply these against the latest PDF currently in `docs/report/Report Draft.pdf`; they supplement the page-oriented guidance above.

- **Pages 1–2, dialogue:** reconcile the abstract's Gemma router/clarification claim with the statement that the dialogue manager was not implemented. The code supports topic clarification, active-topic tracking, and an “anything else?” request for another accepted passage. Describe this limited behavior; do not call it unrestricted dialogue.
- **Page 1, safety:** replace “checks keep reversed meanings from reaching the user.” The documented reversal was flagged and the original is the default after a failed check. This is an observed example, not a guarantee; failed rewrites can be inspected.
- **Page 4, DeBERTa settings:** the latest draft says batch 16 and learning rate 2e-5. The supplied notebook specifies per-device train batch 8, accumulation 4 (effective 32), inference batch 16, and learning rate 1e-5. Verify the actual deployed run config/log before correcting; do not conflate inference batch with training batch.
- **Page 4, commands:** remove unverified `src/...` and Streamlit commands. They do not match the repository tree reviewed; the app uses FastAPI. Use commands from `deberta-docker/README.md`, `gemma-eval-docker/README.md`, and `jargonaut-app/README.md`, and state the external model/data/cache dependencies and offline loading requirement.
- **Pages 1, 5, 8, precision:** qualify 97.4% as answerability precision (370/380); separately give accepted-span overlap precision 95.5% (363/380). Keep answerability recall 83.5% (370/443) distinct from span-hit recall 81.9% (363/443). Apply this to tables and the abstract.
- **Page 2, annotation:** “generated and cross-verified automatically” does not identify annotators/tools. Confirm provenance with the team. Prior evaluation notes identify two Claude agents; state actual tools accurately and do not imply human or independent human validation. Agreement is not proof of correctness.
- **Page 7, readability:** 8.3 is the mean generated-rewrite FKGL before the gate, including outputs later withheld. Do not present it as the reading grade of the displayed answers.
- **Page 7, manual review:** reconcile “40 answers fully read” with “10 shown rewrites read in full.” State actual sample, method, and reviewer, or remove the unverified broad conclusion.
- **Page 8, miss count:** correct “Governing Law accounts for 8 of 10 misses” to “8 of 9 total misses” or “8 of 10 Governing Law cases.”
- **Page 8, routing:** 39/40 includes one expected-topic choice after clarification; direct routing is 38/40. State which denominator/definition is used and keep this separate from the 110-item router benchmark.
- **Page 9, cap example:** it cannot simultaneously preserve every condition and omit “Subject to Section 4.” Acknowledge that the dropped cross-reference may point to exceptions.
- **Pages 9–10, references/contributions:** remove duplicate Platt citation; add/verify the Gemma 4 E4B and NLI model references; retain ContractNLI only if actually used; remove the blanket claim that all references are verified unless confirmed. Confirm individual contributions with the team.

## RoBERTa baseline feasibility and reporting

- A proposed baseline is a useful same-task comparison: it shows whether CUAD-finetuned DeBERTa-v3-large improves on a smaller RoBERTa-base extractor. It does not compare the whole chatbot or the Gemma simplifier.
- The DeBERTa notebook already has a `MODEL` switch for `roberta-base`, and its run name is generated from that value. This suggests the baseline is a modest code change, but running it still requires the external shared dataset/BM25 files, GPU environment, training time, and careful verification. Those data/run folders were not present in the reviewed checkout, so a baseline cannot be run or results claimed from this folder alone.
- If the team runs it, hold the category set, train/validation/test split, chunking, examples/negatives, epoch count and evaluation procedure fixed; train RoBERTa on training only; tune its abstention threshold on validation; then evaluate the selected checkpoint once on untouched test. Report answerability precision/recall and span metrics separately, alongside EM/F1 and retrieval coverage. Label this as a baseline experiment and record the actual run/config/log.
- Do not simply swap the model name and reuse the DeBERTa threshold: thresholds are model-specific and must be tuned on validation. Confirm tokenizer/sequence handling and that both runs use comparable training data.
- With a 1–2 day deadline, first establish that the data and GPU are available and estimate the full run time. If those prerequisites are not ready, state the baseline was proposed but not completed and keep it as future work. Never fabricate a baseline result or select settings based on test scores.
