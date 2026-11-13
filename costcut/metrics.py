"""Per-question metrics. Two are GATED (see gate.py); the rest are reported only.

GATED
  answer_f1      HotpotQA/SQuAD token-F1 of the answer vs the gold answer (yes/no/noanswer must
                 match exactly, as in the official HotpotQA script). Range 0..1.
  bundle_recall  fraction of the gold supporting titles present in the context bundle the
                 synthesis model actually received. Range 0..1. This is the retrieval metric that
                 a context budget can hurt, so it is the one the gate watches.

REPORTED
  answer_em      exact match after normalisation.
  ndcg10         NDCG@10 of the fused retrieval ranking against the gold titles (binary
                 relevance; a title counts once, its first occurrence; ideal DCG = all gold titles
                 at the top). Does not depend on the context budget.
  cite_recall    fraction of gold titles among the passages the answer cited.
  intent_acc     predicted intent == HotpotQA's `type` field.
"""
from __future__ import annotations

import json
import math
import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def ndcg_at_k(ranked_titles: list[str], gold: list[str], k: int = 10) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for r, t in enumerate(ranked_titles[:k], 1):
        if t in gold_set and t not in seen:
            dcg += 1.0 / math.log2(r + 1)
        seen.add(t)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold_set), k) + 1))
    return dcg / idcg


def recall(titles: list[str], gold: list[str]) -> float:
    g = set(gold)
    return len(g & set(titles)) / len(g) if g else 0.0


def parse_json(text: str) -> dict:
    """Lenient: first {...} object in the text. Returns {} if none parses."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


GATED = ("answer_f1", "bundle_recall")
REPORTED = ("answer_em", "ndcg10", "cite_recall", "intent_acc")
ALL = GATED + REPORTED
