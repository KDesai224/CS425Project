# Presentation revision instructions

Use this slide-by-slide edit brief for the 19-slide PDF draft. There was no editable PowerPoint file in the project at review time. Preserve the visual theme where possible, but update the source deck and export a new PDF after edits. Course limit: 25 minutes. Aim for roughly 18–20 minutes of planned delivery plus questions/demo buffer; rehearse the live demo and prepare a backup.

## Global corrections

- Remove all `[TODO]` text, add team names and date, and use consistent capitalization for DeBERTa, Gemma 4 E4B, Run 6, CUAD, and SimpleGAL (after verifying the exact dataset identity).
- Make the central claim: the app retrieves and cites contract evidence, then attempts a simpler rewrite; abstention and checks reduce some risks, but routing and lexical/NLI heuristics remain imperfect.
- Distinguish answerability from span accuracy everywhere: 97.4% answerability precision; 95.5% accepted-span overlap precision; 83.5% answerability recall; 81.9% span-hit recall on CUAD test PDFs.
- Label every result with dataset, split, sample size, and denominator. Keep the 20-clause validation, 102-contract CUAD test, 110-question routing test, and fictional demonstration separate.
- Show 8.3 FKGL as generated rewrite score before gating, not final displayed answer quality. The approximate mean for displayed outputs is 13.33 when original clauses replace withheld rewrites.
- No safety guarantee: state that the checker caught the shown reversed-right example and that flagged rewrites default to the source clause; users may inspect the failed rewrite.
- Add citations/source notes on the final reference slide, including datasets, model cards and metric sources actually used.

## Slide-by-slide edits

### 1. Title

- Replace `[TODO] Team G1T5 · names · date` with confirmed full team names, course/group, and presentation date.
- Keep title/subtitle. Verify name order/spelling with team records.

### 2. Motivation

- Keep the comprehension gap motivation.
- Add one concrete example or short user journey only if it fits cleanly; avoid unsupported claims about prevalence or harm.

### 3. Goal and scope

- Retain research question and ten CUAD-topic scope.
- Add a one-line boundary: the prototype routes to one of ten topics and then searches for the standard category evidence; it does not provide unrestricted legal advice.

### 4. Datasets

- Replace generic “20 labelled clauses” with exact source and split: SimpleGAL validation, 20 clauses, for rewrite comparison; Run 6 has no reported result on the held-out simplifier test set.
- Add CUAD PDF test: 102 contracts / 1,020 category questions / 443 evidence-bearing.
- Add fictional set: 10 invented contracts / 100 contract-topic pairs / 60 natural-language questions, marked demonstration-only.
- Remove the TODO about dataset and ContractNLI only after verifying provenance. Note that the exact 886 synthetic-pair source/tool needs to be documented. If ContractNLI was not used, say so; if it was used in Gemma training, say there.

### 5. Pipeline

- The seven-stage diagram is broadly accurate. Clarify route selects a category; BM25 and DeBERTa use that category’s fixed CUAD question.
- State that the answer includes original passage, section/page and confidence, and that low DeBERTa scores abstain.
- Add a small qualifier that clause expansion adds nearby text but does not fetch referenced definitions.

### 6. Models and settings

- Remove TODO. Show known settings only: DeBERTa 3 epochs, lr 1e-5, seed 42, max length 512/stride 128, 3 hard negatives, no synthetic extractor QA; Gemma E4B + Run 6 LoRA rank 16/lr 2e-4/886 pairs; BM25 top 10; NLI threshold 0.7. Verify training config before including remaining values.
- Explain proposal deviation in one sentence: 31B QLoRA was planned with E4B fallback; the app uses E4B LoRA. Give the actual resource reason only when confirmed.
- Say RoBERTa baseline was not run if verified. Do not imply a comparison.

### 7. Contribution

- Keep contribution claims tied to measurable work: integrated upload-to-answer pipeline, page/heading traceability, router/clarification, calibration, abstention, checks, and evaluation.
- Distinguish new integration from reused components; current code has six parity checks against team outputs.

### 8. CUAD evidence results

- Replace current headline or accompanying text so 97.4% is explicitly answerability precision. Add accepted-span overlap precision 95.5% (363/380), answerability recall 83.5% (370/443), span-hit recall 81.9% (363/443), and retrieval top-10 90.1% of answerable questions.
- Define “accepted”, “answerability”, and “span hit” in a legible footnote or spoken explanation.

### 9. Category results

- Preserve values after recomputing against `jargonaut-app/eval/end_to_end/summary.json`.
- Give positive denominator counts alongside recall if legibility permits. Call weak-category rates noisy where n<35.
- State that 8 of the 9 total misses were Governing Law (or 8/10 Governing Law cases), not “8 of 10 misses.”

### 10. Confidence calibration

- Label probability as calibrated probability that best extracted span overlaps gold evidence, not that the answer is legally correct.
- Keep ECE=0.039 and Brier=0.045 as test-set measures; note only >=0.8 is labelled High because midrange bins are overconfident.

### 11. Routing comparison

- Keep 110-question held-out router comparison, state 80 in-scope / 30 out-of-scope, and clarify test questions were written blind from category descriptions.
- Add fallback latency only if saved result/source supports a number; otherwise remove `[TODO]` and report “not measured.”
- Keep error examples (deadline, email notice, most-favoured pricing). State router results do not equal end-to-end evidence correctness.

### 12. Controlled simplification validation

- Label explicitly “SimpleGAL validation, n=20; not test.”
- Keep SARI/FKGL/NLI and number precision/negation. Add number recall (0.74 zero-shot, 0.75 Run 6), because 1.00 number precision does not mean all source quantities were retained.
- Explain Run 6 trades lower SARI and slightly higher FKGL for higher NLI coverage and observed negation retention; confidence intervals are wide with 20 clauses.
- Remove the TODOs about seeds, test result, and manual sheet only by accurately stating outstanding status. Do not imply those are complete.

### 13. Full pipeline rewrites

- Keep n=380 accepted extraction cases. State 204/380 (53.7%) pass all three checks; 176/380 show the source clause by default.
- Label FKGL 18.9 to 8.3 as clause versus *generated rewrite before gating*. Add ~13.33 as mean displayed output only if calculation and averaging method are confirmed in final source.
- Say NLI is the only failing check for 138 of 176 withheld; clarify number/negation denominators.

### 14. Checker example

- Keep the reversed termination example, but phrase: “The NLI coverage check flagged this rewrite; the app defaulted to the original clause.”
- Add caveat: computed contradiction/support scores are not currently part of the pass/fail gate, so this example does not establish general protection.
- Since failed rewrite is inspectable in the UI, do not say it cannot reach the user.

### 15. Fictional test set

- Replace the TODO with all key counts: 19/19 accepted answers on evidence-bearing pairs (100% answerability precision, 19/28 recall); 0/72 false accepts; 10/10 target clauses hit.
- Add 95% Wilson interval for precision (83–100%) and recall (49–82%) or state intervals in speaker notes.
- Caveat: 10 invented contracts, two chunks each, top-10 retrieval; demonstrates extraction/abstention on designed cases and does not establish real-world generalization or retrieval quality.

### 16. Error analysis

- Keep what worked and what did not, and add clear evidence counts: 8/10 fictional Governing Law clauses missed under active “law governs” wording; near-perfect rate on CUAD-style wording; short clauses pass NLI threshold less often (44% under 80 words vs 88% at 150–300 words).
- Avoid treating the fictional wording diagnostic as a large benchmark. Show it as an illustrative error analysis.
- Mention the routing errors if space allows.

### 17. Limitations and next steps

- Keep small sample sizes, validation-only Run 6, calibration and checker limitations.
- Add no RoBERTa baseline, incomplete reproducibility assets, missing referenced-definition retrieval, and generator/annotation provenance.
- Order next steps by impact: verify provenance/course policy; package assets/run instructions; human review with a defined rubric; improve quantity/party/exception checks; test natural-language query handling; add the baseline if feasible.

### 18. Demo

- Replace `[TODO] Insert app URL / backup recording` with working demo location if shareable and a prepared fallback. Never put private credentials or local machine paths on a public slide.
- Demo one successful grounded answer, one abstention or clarification, and show the original evidence/page. Keep under 3 minutes and rehearse offline fallback.

### 19. Closing and references

- Replace repository/contact TODO with the repository location and group contact if appropriate.
- Add a separate references slide if required, while remaining under 25 minutes. Cite CUAD, DeBERTa, Gemma 4, LoRA, BM25, NLI model, SimpleGAL source, SARI and FKGL sources actually used.
- Finish with the balanced claim: the system can surface traceable evidence and often simplify it, while category limits, wording sensitivity, and imperfect checks still matter.

## Visual and delivery checks

- Check every slide after editing: no overflow, clipped labels, tiny denominator text, or unresolved TODOs.
- Slide 5 currently has a readable seven-step flow, but verify line lengths remain balanced after clarifying the fixed CUAD question.
- Slide 11 latency TODO must not remain.
- Keep decimal values intact and readable; align slide metrics with the final report tables.
- Rehearse against the 25-minute maximum and prepare responses on provenance, baseline omission, model/data deviations, checker limitations, and reproducibility.
