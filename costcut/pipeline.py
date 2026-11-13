"""The pipeline under test, per question:
    1. intent classification   (LLM)  -> "bridge" | "comparison"
    2. query generation        (LLM)  -> {"queries": [...], "entities": [...]}
    3. retrieval               (BM25 + RRF over ~990 paragraphs, local, $0)
    4. grounded synthesis      (LLM)  -> {"answer": ..., "citations": [...]}
"""
from __future__ import annotations

from dataclasses import dataclass

from . import metrics as M
from . import prompts as P
from .config import MAX_TOKENS, Config
from .context import Retriever, build_bundle
from .data import Data, Question
from .llm import LLM
from .pricing import Usage


@dataclass
class Pipeline:
    cfg: Config
    data: Data
    retriever: Retriever
    llm: LLM

    def run(self, q: Question, tag: str) -> dict:
        c, llm = self.cfg, self.llm
        calls = {}

        r1 = llm.call(step="intent", model=c.intent_model, system_blocks=P.system_blocks(P.INTENT_SYSTEM, c.caching),
                      user=P.intent_user(q.question), max_tokens=MAX_TOKENS["intent"], tag=tag)
        calls["intent"] = r1
        intent = str(M.parse_json(r1.text).get("intent", "")).strip().lower()
        if intent not in ("bridge", "comparison"):
            intent = "bridge"  # documented fallback; counted wrong if gold is comparison

        r2 = llm.call(step="qgen", model=c.qgen_model, system_blocks=P.system_blocks(P.QGEN_SYSTEM, c.caching),
                      user=P.qgen_user(q.question, intent), max_tokens=MAX_TOKENS["qgen"], tag=tag)
        calls["qgen"] = r2
        qj = M.parse_json(r2.text)
        queries = [s for s in qj.get("queries", []) if isinstance(s, str) and s.strip()][:3]
        qgen_ok = bool(queries)
        if not queries:
            queries = [q.question]  # documented fallback: search with the raw question

        ranked = self.retriever.search(queries)
        bundle = build_bundle(self.data.docs, ranked, c.budget, c.per_source_cap, c.dedup)

        r3 = llm.call(step="synth", model=c.synth_model, system_blocks=P.system_blocks(P.SYNTH_SYSTEM, c.caching),
                      user=P.synth_user([(t, x) for _, t, x in bundle.docs], q.question),
                      max_tokens=MAX_TOKENS["synth"], tag=tag)
        calls["synth"] = r3
        sj = M.parse_json(r3.text)
        answer = str(sj.get("answer", "")).strip()
        cites = [int(x) for x in sj.get("citations", []) if isinstance(x, (int, float)) or str(x).isdigit()]
        cited_titles = [bundle.docs[i - 1][1] for i in cites if 1 <= i <= len(bundle.docs)]

        ranked_titles = [self.data.docs[i].title for i in ranked]
        bundle_titles = [t for _, t, _ in bundle.docs]
        usage_total = Usage()
        for r in calls.values():
            usage_total = usage_total + r.usage
        return {
            "idx": q.idx, "id": q.id, "gold_answer": q.answer, "gold_type": q.qtype, "gold_titles": q.gold_titles,
            "intent": intent, "queries": queries, "qgen_parsed": qgen_ok,
            "ranked_titles": ranked_titles[:10], "bundle_titles": bundle_titles,
            "bundle_docs": len(bundle.docs), "bundle_est_tokens": bundle.est_tokens,
            "dropped_dupes": bundle.dropped_dupes, "truncated": bundle.truncated,
            "answer": answer, "synth_parsed": bool(sj), "citations": cites,
            "metrics": {
                "answer_f1": M.f1(answer, q.answer),
                "bundle_recall": M.recall(bundle_titles, q.gold_titles),
                "answer_em": M.em(answer, q.answer),
                "ndcg10": M.ndcg_at_k(ranked_titles, q.gold_titles, 10),
                "cite_recall": M.recall(cited_titles, q.gold_titles),
                "intent_acc": float(intent == q.qtype),
            },
            "calls": {k: {"model": r.model, "usage": r.usage.to_dict(), "usd": r.cost, "reused": r.reused,
                          "stop_reason": r.stop_reason} for k, r in calls.items()},
            "usage": usage_total.to_dict(),
            "usd": sum(r.cost for r in calls.values()),
        }
