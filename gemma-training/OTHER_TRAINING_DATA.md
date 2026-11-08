# Other training data for the rewriter

Checked 6 Nov 2026.

The question: could an existing dataset give Model 2 (the Gemma 4 E4B rewriter) better training targets than SimpLegal-Pref and the synthetic pairs? Each dataset was considered both as is and after preprocessing.

**Answer: no.** No available dataset pairs contract clauses with plain-English rewrites that keep the whole clause.

SimpLegal-Synth can be filtered down to roughly 250 usable pairs. But they are the same kind of target the team already has: another model's rewrites, picked by the same checks. Better targets have to be built; see [What to do instead](#what-to-do-instead).

## What a better target needs

Across the runs in `TRAINING_PROGRESS.md`, the fine-tuned model ends up with the word length of its training targets:

| Training targets | Pairs | Targets: syllables per word | Model on validation: syllables per word | Model on validation: FKGL | Model on validation: NLI coverage |
|---|---|---|---|---|---|
| Human SimpLegal Basic (Run 1) | 139 | 1.42 | 1.39 | 7.87 | 0.32 |
| Synthetic, strict subset (Run 5) | 147 | 1.57 | 1.58 | 9.72 | 0.73 |
| Synthetic, all (Run 6) | 886 | 1.63 | 1.60 | 8.74 | 0.79 |
| *Zero-shot Gemma, no fine-tuning* | | | *1.51* | *7.96* | *0.73* |

- **Neither existing target set teaches both qualities.** The human rewrites use plain words but drop content. The synthetic ones keep content but use longer words than zero-shot Gemma.
- **Word length explains Run 6's test gap.** On the test set, Run 6 averages 1.64 syllables per word against zero-shot's 1.52. That accounts for about 1.3 of the 1.34-grade FKGL gap; sentence length is the same (15.4 words for both, averaged over clauses).

A target set that can beat zero-shot therefore needs four things:

1. **Contract clauses**, ideally from the app's 10 CUAD topics.
2. **The whole clause kept.** Every number and negation must pass step 8's checks, and NLI coverage must be at least 0.7.
3. **Plain words**, meaning fewer syllables per word than zero-shot's 1.51, not just short sentences.
   - Run 5's targets reached FKGL 7.79 with short sentences (12.5 words), but their words stayed long.
   - The model trained on them copied the word length but not the short sentences, and its rewrites came out at FKGL 9.72.
4. **No leakage** into the team's SimpLegal validation and test paragraphs, or into the CUAD test contracts used by the app's end-to-end eval.

Reference points for the targets, scored with the team's metrics (`jargonaut/metrics.py`):

| Target set | Pairs | NLI coverage | FKGL | Syllables per word | Words per sentence | Compression |
|---|---|---|---|---|---|---|
| SimpLegal Basic, train, kept by step 9 | 87 | 0.42 | 8.37 | 1.42 | 18.6 | 0.57 |
| SimpLegal Basic, validation references | 20 | 0.30 | 9.04 | 1.38 | 21.5 | 0.50 |
| Synthetic, all | 886 | 0.80 | 8.73 | 1.63 | 12.9 | 0.85 |
| Synthetic, strict subset | 147 | 0.92 | 7.79 | 1.57 | 12.5 | 0.79 |
| *Zero-shot Gemma rewrites of validation* | 20 | *0.73* | *7.96* | *1.51* | *14.6* | *0.75* |

## Datasets checked

| Dataset | What it is | Verdict |
|---|---|---|
| SimpLegal-Synth, test split | 1,204 preference pairs from the SimpLegal authors | **Never use:** it contains the team's test, validation and training paragraphs |
| SimpLegal-Synth, train split | 59,373 preference pairs over 4,700 contract paragraphs | Yes as inputs; mostly no as targets |
| SimpLegal AllGenerationData (Basic) | 16 model rewrites and a "Gold" column for each of 201 paragraphs | No: the same paragraphs as SimpLegal-Pref, and the columns don't reliably line up |
| SIMPLE-LAW (LREC 2026) | 6,230 GPT-3.5 rewrites of mostly legislation | No: the rewrites read at grade 15 and are half the source length |
| Plain English Summarization of Contracts (TL;DRLegal, TOS;DR) | A few hundred contract and terms-of-service sections with plain-English summaries | No: summaries drop content by design |
| BillSum | US bills with their official summaries | No: summaries |
| 2007 restyled Federal Rules of Civil Procedure | Court rules rewritten "stylistic only" | No: the meaning is kept, but the text is still legal prose, and these are not contracts |
| plainlanguage.gov word list | Plain substitutes for formal words | Not training data, but useful as a vocabulary check |
| General simplification sets (Wikipedia, news) | Not legal text | Not tested |

### SimpLegal-Synth

It sits in `GSLD/Data/SimpLegal-Synth` of the SimpLegal repository, the same repository SimpLegal-Pref comes from. The authors built it "by pairing each rejected simplification with a randomly sampled accepted simplification from the same source paragraph". Each row has `original`, `preferred` and `not-preferred` fields. The Basic-grade files were checked.

**Size.**
- The train split has 59,373 rows over 4,700 paragraphs, a median of 13 rows per paragraph. That makes 14,288 distinct pairs of a paragraph and a preferred rewrite.
- The test split has 1,204 rows over 1,179 paragraphs.
- No paragraph appears in both splits.

**Leakage.** Paragraphs were matched on normalised text. Near-duplicates were found by the share of each paragraph's 10-word phrases that also appear in Synth.

| Team split | Found word for word in Synth test | Near-duplicate in Synth train (at least 50% of 10-word phrases shared) |
|---|---|---|
| SimpLegal train (139) | 132 | 1 |
| SimpLegal validation (20) | 18 | 1 |
| SimpLegal test (40) | 38 | 0 (largest share 22%) |
| `pipeline_test` (100) | 0 | 0 |

- The test split holds almost all of SimpLegal-Pref, so training on it would mean training on the team's test set.
- The train split is clean apart from the two near-duplicates.
- One Synth train paragraph is at least 80% contained in a CUAD test contract, which is part of the app's end-to-end set. Nine are contained in CUAD train contracts.

**Quality as targets.** Each distinct pair was run through two sets of checks:
- step 8's checks from `scripts/08_filter_augmentation.py`: every number kept and none invented, negation kept, FKGL at most 11, compression between 0.35 and 1.1
- two added checks: chatbot leftovers (such as "Here is…" or "(Note: …)") and repeated sentences

Each pair is counted under the first check it fails:

| Result | Preferred rewrites (14,288) | Not-preferred rewrites (55,465) |
|---|---|---|
| Pass | 20.3% | 16.2% |
| Chatbot leftovers | 20.6% | 18.5% |
| Too long (compression above 1.1) | 18.1% | 18.9% |
| Number dropped | 18.1% | 18.4% |
| Number invented | 9.0% | 7.6% |
| Too short (compression below 0.35) | 5.3% | 6.0% |
| Repeated sentences | 5.2% | 6.9% |
| Too hard (FKGL above 11) | 1.8% | 5.6% |
| Negation lost | 1.6% | 1.9% |

- **The preference label barely tracks the team's criteria.** Preferred rewrites pass only slightly more often than not-preferred ones.
- **Passing paragraphs.** 1,842 of the 4,700 paragraphs have at least one preferred rewrite that passes. The passing pairs average FKGL 8.55 and compression 0.70.
- **Completeness of the passing rewrites.** A random 150 of those paragraphs, one passing rewrite each, were scored with the team's NLI model:
  - Mean coverage was 0.52 and the median 0.55.
  - 37% reach coverage 0.7, and 13% reach both coverage 0.7 and FKGL 8.
  - Across the whole split, that is about 250 paragraphs, with a 95% interval of roughly 160 to 360.
  - Their word length was not measured.

**Verdict.**
- **As targets:** the usable remainder is a few hundred pairs of the same kind as the team's synthetic set. It would make a filtered set bigger without changing what the model learns.
- **As inputs:** the 4,700 paragraphs, 4,397 of them between 40 and 300 words long, are a ready supply of contract text for generating new targets. Drop the two near-duplicates and the CUAD-test paragraph first.

The repository also has the authors' LoRA DPO script, `GSLD/Scripts/DPO_trainer.py`, which is a starting point for a DPO run. Its settings:
- 4-bit base model
- LoRA rank 32, alpha 16, dropout 0.05
- beta 0.1, learning rate 2.5e-5, 1 epoch
- effective batch 32, maximum length 1,024 tokens

### SimpLegal AllGenerationData (Basic)

`GSLD/Data/AllGenerationData/Basic_grade_generatedSimplifications.xlsx` has 201 rows, the same number of paragraphs as SimpLegal-Pref. Each row has an `Original`, a `Gold` and 16 model rewrites (`SimplifiedResponse-1` to `-16`).

- **Coverage.** 111 originals match a team paragraph word for word: 73 train, 11 validation and 27 test. So the file offers at most extra candidate targets for the training paragraphs. Using anything for validation or test paragraphs would leak.
- **The `Gold` column is not the SimpLegal-Pref Basic reference the team uses.** For the 111 matched paragraphs, the team's reference was found in these places:
  - in the same row's Gold: 1
  - among the same row's 16 rewrites: 1
  - in a neighbouring row's Gold: 6
  - nowhere: 103
- **Some entries describe a different clause.**
  - Row 2's original is an "Entire Agreement" clause, but its Gold is about New York law and arbitration.
  - One of row 1's rewrites of a notices clause is about indemnification.
- It was not scored further.

### SIMPLE-LAW

SIMPLE-LAW comes from the paper "Decode the Law" (LREC 2026). It has 6,230 pairs:
- about 3,000 Indian sections and acts, 1,500 US laws and 1,500 UK laws
- rewritten by GPT-3.5-Turbo using few-shot and chain-of-thought prompts built from 50 expert-written examples

The paper's own figures, from its text and Table 1:

| | Original | Simplified |
|---|---|---|
| Average words | 292.7 | 137.2 |
| FKGL | 23.56 | 15.08 |
| Dale–Chall (vocabulary-based) | 9.83 | 9.64 |

The rewrites are under half the source length, still read at university level, and barely change the vocabulary. Training on them would repeat Run 1's content loss at a harder reading level. Ruled out without downloading.

### Plain-English summaries: TL;DRLegal, TOS;DR and BillSum

- **Manor and Li (2019)** pair a few hundred contract and terms-of-service sections with plain-English summaries taken from TL;DRLegal and TOS;DR.
- **BillSum** pairs US bills with their official summaries.

Summaries leave content out by design. That is the failure Runs 1 to 4 already show, so neither set fits. Neither was downloaded.

### 2007 restyled Federal Rules of Civil Procedure

The committee notes say the 2007 restyling was "intended to be stylistic only", so the old and new rules form pairs that keep the meaning. But the new text is still formal legal prose, not grade-8 English, and these are court rules, not contract clauses. Not used.

### plainlanguage.gov word list

The US government's "Use simple words and phrases" list gives plain substitutes for formal words, for example "commence" → "start", "terminate" → "end" and "utilize" → "use". It is not training data. It does give the direct vocabulary check that Run 6 lacked: count listed words in a rewrite, or prefer candidates that use none.

### General simplification datasets

Simplification sets built from Wikipedia or news were not tested, because they are not legal text.

## Inputs for building new targets

No dataset supplies the targets, but several supply clauses to rewrite:

| Source | Clauses | Notes |
|---|---|---|
| CUAD train, the app's 10 topics | 3,445 labelled spans in 384 of the 408 contracts (median 38 words) | The extractor's training contracts, so nothing comes from the app's CUAD test set |
| SimpLegal-Synth train | 4,700 paragraphs | Drop the 2 near-duplicates of team paragraphs and the 1 CUAD-test paragraph |
| The synthetic set's own sources | 886 (624 CUAD, 262 ContractNLI) | A new target can be compared with Claude's on the same clause |

- **CUAD spans per topic:** Cap On Liability 554, Anti-Assignment 517, Insurance 443, Expiration Date 384, Governing Law 374, Exclusivity 332, Ip Ownership Assignment 257, Termination For Convenience 205, Non-Compete 200, Renewal Term 179.
- **The current synthetic set barely uses these topics.** Only 242 of its 805 CUAD candidates came from them, and 192 of the 624 accepted pairs.

## What to do instead

Build targets that keep the whole clause and use plain words. Compare them at the same size, about 150 pairs (Run 5's size), so that only the choice of targets changes:

1. **Claude's synthetic pairs, filtered for plain words.** 150 have NLI coverage of at least 0.7 and at most 1.55 syllables per word. They average 1.47 syllables per word, FKGL 7.27 and coverage 0.92. No generation is needed.
2. **Gemma's own best rewrites.** Sample 4 zero-shot rewrites per clause and keep the best one: every number and negation kept, NLI coverage of at least 0.8, and the shortest words.
3. **A control:** Claude's rewrites of the same clauses as set 2.
4. **DPO on set 2's samples.** Pair the kept rewrite against a failed sample or a hand-made error: a flipped "may" or "must", a dropped "unless", a moved deadline, a changed number or swapped parties.

Compare all of them with zero-shot, Run 5 and Run 6 on two sets: the SimpLegal test set and the 380 clauses from the app's end-to-end eval.

Set 2 is picked with the same NLI model the app's checks use, `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, so it has an advantage on that measure. Check a sample of the rewrites by hand.

## How this was checked

- **Scripts.** All numbers come from one-off scripts run on 6 Nov 2026. The scripts are not saved in this project.
- **NLI coverage.** This uses `metrics.NLIScorer` with the team's model: the mean probability that the rewrite entails each source sentence. The SimpLegal-Synth sample was scored on CPU.
- **Readability.** FKGL, syllable and sentence counts use textstat, as `metrics.py` does.
- **Run figures.** Run results come from `outputs/predictions/simplegal_val/`, `outputs/metrics/` and `TRAINING_PROGRESS.md`.

## Sources

- SimpLegal repository (SimpLegal-Pref, SimpLegal-Synth, AllGenerationData, DPO script): <https://github.com/GSLD-SimpLegal/FromJargonToClarity>
- "Decode the Law" (SIMPLE-LAW), LREC 2026: <https://aclanthology.org/2026.lrec-1.45/>. Data and code: <https://github.com/mohammeddanishrabbani/Legal-Simplification-mrabbani>
- Manor and Li (2019), Plain English Summarization of Contracts: <https://aclanthology.org/W19-2201/>
- Kornilova and Eidelman (2019), BillSum: <https://aclanthology.org/D19-5406/>
- Federal Rules of Civil Procedure, Rule 1, committee note on the 2007 amendment: <https://www.law.cornell.edu/rules/frcp/rule_1>
- plainlanguage.gov, "Use simple words and phrases" (archived copy): <https://webarchive.library.unt.edu/web/20121014020023mp_/http://www.plainlanguage.gov/howto/wordsuggestions/index.cfm>
- Hendrycks et al. (2021), CUAD: <https://arxiv.org/abs/2103.06268>
