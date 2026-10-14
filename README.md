# Jargonaut

**Evidence-Grounded Contract Simplification**  
CS425 Project | Group G1T5

## Overview

Jargonaut is a proposed task-oriented chatbot that helps users understand contracts and agreements. Users will upload a contract and ask questions about its contents. The chatbot will retrieve relevant evidence and explain it in plain English, alongside the original passage and source reference.

The initial scope covers selected topics in English commercial contracts.

**Research question:** Does combining evidence extraction with a separately trained simplification component improve understandability while preserving contractual meaning?

## Planned Approach

1. **Document processing and retrieval:** Split contracts into overlapping, clause-based passages while preserving headings and page references. Use BM25 to retrieve the top 10 candidate passages.
2. **Evidence extraction:** Fine-tune DeBERTa-v3-large to identify relevant answer spans. Use an answerability threshold to return an insufficient-evidence response when support is inadequate. RoBERTa-base will serve as the baseline.
3. **Plain-English simplification:** Fine-tune Gemma using LoRA/QLoRA to simplify extracted evidence while preserving obligations, conditions, exceptions, and other contractual details.

## Datasets

| Dataset | Planned use |
| --- | --- |
| CUAD | Train evidence extraction using annotated contract clauses, with plain-language Q&A augmentation and negative examples. |
| ContractNLI | Extend evidence extraction to confidentiality-related concepts. |
| SimpLegal-Pref | Train simplification using checked Basic-level legal-text-to-plain-English pairs. |

## Evaluation

- **Retrieval:** Recall@k.
- **Evidence extraction:** Exact Match, F1, and answer precision/recall.
- **Simplification:** Flesch–Kincaid, SARI, and faithfulness checks.

## Repository Structure

```text
CS425Project/
├── Notebooks/                 # Reserved for project notebooks
├── ProjectProposal_G1T5.pdf    # Project proposal
└── README.md
```

## Project Status

The repository currently contains the project proposal and an empty notebook directory. Implementation, dependencies, and setup instructions will be added as development progresses.

## Team

| Members | Responsibilities |
| --- | --- |
| Kiara, Ridheema | BM25 retrieval and evaluation; DeBERTa training and evaluation |
| Ian, Shernice | Gemma simplification training and evaluation |
| Theo | Data preparation, augmentation, and frontend |

All members will contribute to error analysis, testing, and end-to-end integration.

## Proposal

See [ProjectProposal_G1T5.pdf](ProjectProposal_G1T5.pdf) for the full project plan.
