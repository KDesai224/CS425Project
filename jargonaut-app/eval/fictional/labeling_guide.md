# Labeling guide: fictional contract test set

Written before any model was run on these contracts. Both annotators followed it.

## What to label

Every contract × every one of the 10 categories (100 pairs), not only the topic a
contract seems to be about. For each pair, decide whether the contract contains
evidence for the category's CUAD question, and if so quote the evidence.

| Category | CUAD question (Details) |
|---|---|
| Governing Law | Which state/country's law governs the interpretation of the contract? |
| Expiration Date | On what date will the contract's initial term expire? |
| Renewal Term | What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice. |
| Termination For Convenience | Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)? |
| Cap On Liability | Does the contract include a cap on liability upon the breach of a party's obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery. |
| Anti-Assignment | Is consent or notice required of a party if the contract is assigned to a third party? |
| Non-Compete | Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector? |
| Exclusivity | Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all "requirements" from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both). |
| Ip Ownership Assignment | Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events? |
| Insurance | Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty? |

## Rules

1. **Evidence establishes the thing asked about**, as in CUAD, where a category is
   labeled only when the clause is present: a governing law is chosen, an end date
   is set, a renewal happens, a party may terminate without cause, liability is
   capped, assignment needs consent or notice, competition is restricted, dealing is
   exclusive, IP is transferred, insurance must be kept.
2. **A sentence that only denies or excludes the thing is not evidence**, for example
   "There is no automatic renewal" or a "nonexclusive license". List such sentences
   as `near_misses`: they are the set's likely traps.
3. **Quote verbatim**, whole sentences, the fewest sentences that answer the question.
   One string per contiguous passage. Matching ignores spaces and punctuation, so
   line breaks and quote styles do not matter.
4. Exceptions and conditions belong in the span only when they are in the same
   sentence or are needed to answer the question (e.g. a permitted-assignment clause
   that still requires notice is Anti-Assignment evidence).
5. If unsure, label your best reading and say why in `note`.

## Files

- `gold_a.jsonl`, `gold_b.jsonl`: the two annotators' labels, one line per doc × category:
  `{"doc", "category", "spans": [...], "near_misses": [...], "note"}`
- `gold.jsonl`: the adjudicated labels the evaluation scores against.
- `questions.jsonl`: plain-English user questions for the routing pass:
  `{"doc", "question", "expected": "<category>" | "none", "note"}`
