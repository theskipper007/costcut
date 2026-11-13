"""Orchestration: k baseline reruns -> noise floor -> each lever -> gate -> results.json + report.txt."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__
from . import metrics as M
from .config import ALL, BASELINE, BUDGETS, COMBINED_BUDGET, MAX_TOKENS, PER_SOURCE_CAP, Config
from .context import CANDIDATES, PER_QUERY_TOP, RRF_K, Retriever
from .data import LICENSE, SOURCE, load
from .gate import ALPHA, FORMULA, POWER, check, noise, paired_bootstrap, passes
from .llm import LLM, THINKING_OFF, BudgetExceeded, ledger_total
from .pricing import BIG, SMALL, Usage, price_table
from .report import render

ROOT = Path(__file__).resolve().parent.parent
LEVER_TAG = "L1"


def git_info() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": g("rev-parse", "HEAD") or None, "dirty": bool(g("status", "--porcelain", "--", "costcut"))}
    except FileNotFoundError:
        return {"commit": None, "dirty": None}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    agg = {"n": n, "metrics": {m: sum(r["metrics"][m] for r in rows) / n for m in M.ALL}}
    agg["usd_per_question"] = sum(r["usd"] for r in rows) / n
    by_step = {}
    for step in ("intent", "qgen", "synth"):
        u = Usage()
        usd = 0.0
        stops: dict[str, int] = {}
        for r in rows:
            c = r["calls"][step]
            u = u + Usage.from_dict(c["usage"])
            usd += c["usd"]
            stops[c["stop_reason"]] = stops.get(c["stop_reason"], 0) + 1
        by_step[step] = {"model": rows[0]["calls"][step]["model"], "usage_total": u.to_dict(),
                         "prompt_tokens_per_q": u.prompt_tokens / n, "output_tokens_per_q": u.output_tokens / n,
                         "usd_per_q": usd / n, "stop_reasons": stops,
                         "cache_read_share": (u.cache_read_input_tokens / u.prompt_tokens) if u.prompt_tokens else 0.0}
    agg["by_step"] = by_step
    tot = Usage()
    for r in rows:
        tot = tot + Usage.from_dict(r["usage"])
    agg["usage_total"] = tot.to_dict()
    agg["bundle_est_tokens_per_q"] = sum(r["bundle_est_tokens"] for r in rows) / n
    agg["bundle_docs_per_q"] = sum(r["bundle_docs"] for r in rows) / n
    agg["dropped_dupes_total"] = sum(r["dropped_dupes"] for r in rows)
    agg["truncated_total"] = sum(r["truncated"] for r in rows)
    agg["parse_failures"] = {"qgen": sum(not r["qgen_parsed"] for r in rows),
                             "synth": sum(not r["synth_parsed"] for r in rows)}
    return agg


def run_config(cfg: Config, data, retriever, llm: LLM, tag: str, workers: int) -> dict:
    from .pipeline import Pipeline
    p = Pipeline(cfg, data, retriever, llm)
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda q: p.run(q, tag), data.questions))
    return {"config": cfg.name, "tag": tag, "agg": aggregate(rows), "per_question": rows}


def run(*, backend: str, n: int, k: int, workers: int, max_usd: float | None, only: list[str] | None,
        out: Path | None, cache: bool = True) -> dict:
    t0 = dt.datetime.now(dt.timezone.utc)
    data = load(n)
    retriever = Retriever(data.docs)
    llm = LLM(backend, max_usd=max_usd, cache=cache)
    ledger_before = ledger_total() if backend == "anthropic" else 0.0
    plan = [(BASELINE, f"b{i}") for i in range(1, k + 1)]
    plan += [(c, LEVER_TAG) for c in ALL[1:] if not only or c.name in only]

    runs, aborted = [], None
    for cfg, tag in plan:
        print(f"[costcut] {cfg.name:18s} tag={tag:3s} ...", flush=True)
        try:
            res = run_config(cfg, data, retriever, llm, tag, workers)
        except BudgetExceeded as e:
            aborted = f"{cfg.name}/{tag}: {e}"
            print(f"[costcut] ABORTED {aborted}", flush=True)
            break
        a = res["agg"]
        print(f"           ${a['usd_per_question']:.5f}/q  F1={a['metrics']['answer_f1']:.3f}  "
              f"recall={a['metrics']['bundle_recall']:.3f}  spent so far ${llm.spend.usd:.4f}"
              f"{'' if backend == 'anthropic' else ' (dry-run projection $%.4f)' % sum(v['usd'] for v in llm.spend.by_model.values())}",
              flush=True)
        runs.append(res)

    results = build_results(runs, data, n, k, backend, llm, ledger_before, max_usd, aborted, t0, workers)
    out = out or ROOT / ("results.json" if backend == "anthropic" else "results.dry.json")
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    rep = render(results)
    rep_path = out.with_name(out.name.replace("results", "report").replace(".json", ".txt"))
    rep_path.write_text(rep, encoding="utf-8")
    print(rep)
    print(f"[costcut] wrote {out.name} and {rep_path.name}")
    return results


def build_results(runs, data, n, k, backend, llm, ledger_before, max_usd, aborted, t0, workers) -> dict:
    import anthropic  # for the SDK version only
    base_runs = [r for r in runs if r["config"] == "baseline"]
    lever_runs = [r for r in runs if r["config"] != "baseline"]
    gate_noise, summary = {}, {}
    if len(base_runs) >= 3:
        for m in M.GATED + M.REPORTED:
            gate_noise[m] = noise(m, [r["agg"]["metrics"][m] for r in base_runs], n).to_dict()
        base_cost_runs = [r["agg"]["usd_per_question"] for r in base_runs]
        base_cost = sum(base_cost_runs) / len(base_cost_runs)
        per_q_base = {m: [sum(r["per_question"][i]["metrics"][m] for r in base_runs) / len(base_runs)
                          for i in range(n)] for m in M.ALL}
        from .gate import sd
        summary["baseline"] = {"usd_per_question": base_cost, "usd_per_question_runs": base_cost_runs,
                               "usd_per_question_sd": sd(base_cost_runs), "saving_vs_baseline": 0.0,
                               "metrics": {m: gate_noise[m]["mean"] for m in M.ALL}, "k": len(base_runs)}
        from .gate import Noise
        for r in lever_runs:
            vs = []
            for m in M.GATED:
                nz = Noise(**gate_noise[m])
                vs.append(check(nz, [r["agg"]["metrics"][m]]))
            c = r["agg"]["usd_per_question"]
            summary[r["config"]] = {
                "usd_per_question": c, "saving_vs_baseline": 1 - c / base_cost,
                "metrics": r["agg"]["metrics"],
                "gate": {"pass": passes(vs), "verdicts": [v.__dict__ for v in vs]},
                "reported_drops": {m: gate_noise[m]["mean"] - r["agg"]["metrics"][m] for m in M.REPORTED},
                "paired_bootstrap": {m: paired_bootstrap(per_q_base[m], [q["metrics"][m] for q in r["per_question"]])
                                     for m in M.GATED},
            }
    spend_this = llm.spend.usd
    return {
        "meta": {"tool": "costcut", "version": __version__, "created_utc": t0.isoformat(timespec="seconds"),
                 "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "git": git_info(), "python": platform.python_version(), "platform": platform.platform(),
                 "sdk": f"anthropic-python {anthropic.__version__}", "backend": backend,
                 "argv": sys.argv[1:], "workers": workers, "aborted": aborted,
                 "dry_run_note": None if backend == "anthropic" else
                 "FAKE BACKEND: every quality number is junk; usd figures are a pessimistic projection"},
        "dataset": {"source": SOURCE, "license": LICENSE, "file_sha256": data.sha256,
                    "questions": f"rows 0..{n - 1} in published order", "n": n,
                    "corpus_docs": len(data.docs), "corpus_rows": data.n_rows_file},
        "design": {"k_baseline_reruns": k, "r_runs_per_lever": 1, "baseline_tags": [f"b{i}" for i in range(1, k + 1)],
                   "lever_tag": LEVER_TAG,
                   "lever_tag_note": "all lever runs share replicate tag L1, so a request that is byte-identical "
                                     "across two lever configs (e.g. the large-tier intent call in the three context "
                                     "configs) is made once and reused; its recorded usage is still charged to each "
                                     "config's cost per question",
                   "retrieval": {"candidates": CANDIDATES, "per_query_top": PER_QUERY_TOP, "rrf_k": RRF_K,
                                 "bm25": {"k1": 1.5, "b": 0.75}},
                   "budgets_est_tokens": list(BUDGETS), "per_source_cap_est_tokens": PER_SOURCE_CAP,
                   "combined_budget": COMBINED_BUDGET, "token_estimate": "ceil(chars/4)",
                   "max_tokens": MAX_TOKENS},
        "models": {"large": BIG, "cheap": SMALL, "thinking": THINKING_OFF,
                   "effort": "API default (not sent)", "sampling": "API default (not sent)"},
        "prices": price_table(),
        "gate": {"formula": FORMULA, "alpha": ALPHA, "power": POWER, "gated_metrics": list(M.GATED),
                 "reported_metrics": list(M.REPORTED), "noise": gate_noise,
                 "preregistered": "README.md section 'Pre-registration' (2026-11-13), committed before any lever ran"},
        "configs": {c.name: c.to_dict() for c in ALL},
        "summary": summary,
        "spend": {"this_invocation_usd": spend_this, "ledger_before_usd": ledger_before,
                  "ledger_total_usd": (ledger_before + spend_this) if backend == "anthropic" else 0.0,
                  "projected_usd_if_real": None if backend == "anthropic" else
                  sum(v["usd"] for v in llm.spend.by_model.values()),
                  "real_calls": llm.spend.real_calls, "reused_calls": llm.spend.reused_calls,
                  "by_model": llm.spend.by_model, "cap_usd": max_usd},
        "runs": runs,
    }
