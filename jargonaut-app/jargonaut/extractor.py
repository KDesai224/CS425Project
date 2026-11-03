"""Step 4: find the best answer span in the retrieved chunks, with a confidence score.

A port of predict() in deBERTa_training.ipynb (Kiara, Ridheema) for one question:
each chunk is paired with the full CUAD question and split into 512-token windows
with a 128-token stride. In each window the best span (up to max_answer_tokens
long, searched over the top 20 start and end logits) is compared with the
no-answer (CLS) score. diff = best span score - null score is the answerability
score, and the window with the highest diff gives the answer. Below the tuned
threshold the system abstains.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .config import CALIBRATION, DEBERTA_RUN, QA_BATCH_SIZE, QA_MAX_LEN, QA_STRIDE, QA_TOPK, deberta_settings


@dataclass
class Span:
    diff: float          # best span score - null score; the confidence score
    text: str
    start: int           # character offsets in the document, end exclusive
    end: int
    chunk_id: str
    chunk_rank: int      # BM25 rank of the chunk it came from (1 = best)


@dataclass
class Extraction:
    best: Span | None
    accepted: bool
    threshold: float
    probability: float | None = None              # calibrated P(span is real evidence), if calibrated
    alternatives: list = field(default_factory=list)  # other accepted spans, best first, not overlapping


class Extractor:
    def __init__(self, run_dir=DEBERTA_RUN, device=None, batch_size=QA_BATCH_SIZE):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = (self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
                         and torch.cuda.is_bf16_supported())
        self.tok = AutoTokenizer.from_pretrained(run_dir / "final", use_fast=True)
        # fp32 weights with bf16 autocast, as in the notebook
        self.model = AutoModelForQuestionAnswering.from_pretrained(run_dir / "final").to(self.device).eval()
        settings = deberta_settings()
        self.threshold = settings["threshold"]
        self.max_answer_tokens = settings["max_answer_tokens"]
        self.batch_size = batch_size
        self.calibration = None
        if CALIBRATION.is_file():
            with open(CALIBRATION, encoding="utf-8") as f:
                self.calibration = json.load(f)

    def probability(self, diff):
        """Platt-scaled probability that a span with this score overlaps real evidence."""
        if self.calibration is None:
            return None
        z = self.calibration["a"] * diff + self.calibration["b"]
        return 1 / (1 + math.exp(-z))

    def window_spans(self, question, candidates, topk=QA_TOPK):
        """Best span of every window, in window order. candidates: [(Chunk, bm25_score)], best first."""
        torch, tok = self.torch, self.tok
        if not candidates:
            return []
        enc = tok([question] * len(candidates), [c.text for c, _ in candidates], truncation="only_second",
                  max_length=QA_MAX_LEN, stride=QA_STRIDE, return_overflowing_tokens=True,
                  return_offsets_mapping=True, padding=True, return_tensors="pt")
        smap, offs = enc.pop("overflow_to_sample_mapping"), enc.pop("offset_mapping")
        S, E = [], []
        with torch.no_grad():
            for i in range(0, len(smap), self.batch_size):
                batch = {k: v[i:i + self.batch_size].to(self.device) for k, v in enc.items()}
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.use_bf16):
                    o = self.model(**batch)
                S.append(o.start_logits.float().cpu()); E.append(o.end_logits.float().cpu())
        S, E = torch.cat(S).numpy(), torch.cat(E).numpy()

        spans = []
        for f in range(len(smap)):
            seq, off = enc.sequence_ids(f), offs[f]
            rank = int(smap[f])
            chunk = candidates[rank][0]
            ctx = np.array([s == 1 for s in seq])
            s_l, e_l = np.where(ctx, S[f], -1e9), np.where(ctx, E[f], -1e9)
            cls = enc["input_ids"][f].tolist().index(tok.cls_token_id)
            null = S[f][cls] + E[f][cls]
            loc, loc_sc = None, -1e9
            for s in np.argsort(s_l)[-topk:]:
                for e in np.argsort(e_l)[-topk:]:
                    if not ctx[s] or not ctx[e] or e < s or e - s + 1 > self.max_answer_tokens:
                        continue
                    if s_l[s] + e_l[e] > loc_sc:
                        loc, loc_sc = (s, e), s_l[s] + e_l[e]
            if loc is None:
                continue
            a, b = int(off[loc[0]][0]), int(off[loc[1]][1])
            if b <= a:
                continue
            spans.append(Span(float(loc_sc - null), chunk.text[a:b], chunk.char_start + a,
                              chunk.char_start + b, chunk.chunk_id, rank + 1))
        return spans

    def extract(self, question, candidates):
        spans = self.window_spans(question, candidates)
        # Highest diff wins; on a tie the earlier window wins, as in the notebook's strict ">".
        ranked = sorted(spans, key=lambda s: -s.diff)
        best = ranked[0] if ranked else None
        accepted = bool(best and best.text.strip() and best.diff >= self.threshold)
        alternatives = []
        if accepted:
            kept = [best]
            for s in ranked[1:]:
                if s.diff < self.threshold:
                    break
                if s.text.strip() and all(s.end <= k.start or s.start >= k.end for k in kept):
                    kept.append(s)
            alternatives = kept[1:]
        return Extraction(best, accepted, self.threshold,
                          self.probability(best.diff) if best else None, alternatives)
