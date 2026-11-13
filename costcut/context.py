"""Retrieval and the context bundle.

Retrieval (same in every configuration): BM25 (k1=1.5, b=0.75) over the corpus, run once per
generated query (top 50 each), fused with reciprocal-rank fusion (RRF, k=60), ties broken by
doc id. The fused list is cut to CANDIDATES = 20 documents.

Bundle, baseline ("generous"): all 20 candidates, whole paragraphs, in rank order. No budget,
no caps, no de-duplication.

Bundle, context-budget lever: walk the same 20 candidates in rank order and
  1. drop a document whose content hash (title + text) was already added    (dedup)
  2. cut the paragraph to at most `per_source_cap` tokens at a sentence boundary (per-source cap)
  3. add it if it fits in what is left of `budget` tokens; otherwise STOP     (hard budget)
Tokens here are ESTIMATED as ceil(chars / 4), the usual cheap local proxy. The real token
counts and the cost come from the API's usage fields, not from this estimate.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .bm25 import BM25
from .data import Doc

CANDIDATES = 20
PER_QUERY_TOP = 50
RRF_K = 60


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class Retriever:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.bm25 = BM25([f"{d.title} {d.text}" for d in docs])

    def search(self, queries: list[str], k: int = CANDIDATES) -> list[int]:
        score: dict[int, float] = {}
        for q in queries:
            s = self.bm25.scores(q)
            # zero-score documents share no term with the query and are not "retrieved"
            ranked = sorted((i for i in range(len(s)) if s[i] > 0), key=lambda i: (-s[i], i))[:PER_QUERY_TOP]
            for rank, i in enumerate(ranked, 1):
                score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(score, key=lambda i: (-score[i], i))[:k]


_SENT = re.compile(r"(?<=[.!?])\s+")


def cap_text(text: str, cap: int) -> str:
    if est_tokens(text) <= cap:
        return text
    out = ""
    for s in _SENT.split(text):
        cand = (out + " " + s).strip() if out else s
        if est_tokens(cand) > cap:
            break
        out = cand
    return out if out else text[: cap * 4]


@dataclass
class Bundle:
    docs: list[tuple[int, str, str]]  # (doc id, title, text as sent)
    est_tokens: int
    dropped_dupes: int
    truncated: int
    stopped_at_budget: bool


def build_bundle(docs: list[Doc], ranked: list[int], budget: int | None, per_source_cap: int | None,
                 dedup: bool) -> Bundle:
    out: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    used = dupes = trunc = 0
    stopped = False
    for i in ranked:
        d = docs[i]
        if dedup:
            h = d.content_hash
            if h in seen:
                dupes += 1
                continue
            seen.add(h)
        text = d.text
        if per_source_cap is not None:
            capped = cap_text(text, per_source_cap)
            trunc += capped != text
            text = capped
        t = est_tokens(f"{d.title}: {text}")
        if budget is not None and used + t > budget:
            stopped = True
            break
        out.append((d.id, d.title, text))
        used += t
    return Bundle(docs=out, est_tokens=used, dropped_dupes=dupes, truncated=trunc, stopped_at_budget=stopped)
