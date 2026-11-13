"""Plain-text report from results.json. Every number printed here is read from that dict."""
from __future__ import annotations


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def render(R: dict) -> str:
    L: list[str] = []
    meta, ds, gate, S = R["meta"], R["dataset"], R["gate"], R["summary"]
    L.append("costcut: what three inference-cost levers really save, under a fixed quality gate")
    L.append("=" * 84)
    if meta.get("dry_run_note"):
        L.append(f"*** {meta['dry_run_note']} ***")
    if meta.get("aborted"):
        L.append(f"*** ABORTED: {meta['aborted']} ***")
    L.append(f"commit {meta['git']['commit']} (dirty={meta['git']['dirty']})  {meta['sdk']}  {meta['finished_utc']}")
    L.append(f"data   {ds['source']}  [{ds['license']}]  {ds['questions']} (n={ds['n']}); "
             f"corpus {ds['corpus_docs']} paragraphs")
    m = R["models"]
    L.append(f"models large={m['large']}  cheap={m['cheap']}  thinking off on both; effort and sampling at API default")
    p = R["prices"]
    L.append(f"prices {p['unit']}, {p['date']}, {p['source']}")
    for name, tiers in p["models"].items():
        t = tiers[0]
        L.append(f"       {name:18s} in {t['input']:<5} write5m {t['cache_write_5m']:<6} read {t['cache_read']:<5} "
                 f"out {t['output']}")
    L.append("")

    if gate["noise"]:
        L.append(f"Noise floor: k={R['design']['k_baseline_reruns']} baseline reruns, r=1 run per lever")
        L.append(f"  {'metric':14s} {'gated':5s} {'mean':>7s} {'runs':24s} {'sigma':>7s} {'MDE':>7s}")
        for name, nz in gate["noise"].items():
            runs = " ".join(f"{x:.3f}" for x in nz["runs"])
            L.append(f"  {name:14s} {'yes' if name in gate['gated_metrics'] else '':5s} {nz['mean']:7.3f} "
                     f"{runs:24s} {nz['sigma_run']:7.4f} {nz['mde']:7.4f}")
        L.append(f"  {gate['formula']}")
        L.append("")

    if S:
        L.append("Cost per question and gate (cost from API usage x price table)")
        hdr = f"  {'config':18s} {'$/question':>11s} {'saving':>8s} {'F1':>6s} {'dF1':>7s} {'recall':>6s} {'drec':>7s} " \
              f"{'nDCG10':>6s} {'gate':>5s}"
        L.append(hdr)
        b = S["baseline"]
        L.append(f"  {'baseline':18s} {b['usd_per_question']:11.6f} {'':>8s} {b['metrics']['answer_f1']:6.3f} {'':>7s} "
                 f"{b['metrics']['bundle_recall']:6.3f} {'':>7s} {b['metrics']['ndcg10']:6.3f} {'':>5s}")
        for name, s in S.items():
            if name == "baseline":
                continue
            v = {x["metric"]: x for x in s["gate"]["verdicts"]}
            L.append(f"  {name:18s} {s['usd_per_question']:11.6f} {_pct(-s['saving_vs_baseline']):>8s} "
                     f"{s['metrics']['answer_f1']:6.3f} {-v['answer_f1']['drop']:+7.3f} "
                     f"{s['metrics']['bundle_recall']:6.3f} {-v['bundle_recall']['drop']:+7.3f} "
                     f"{s['metrics']['ndcg10']:6.3f} {'PASS' if s['gate']['pass'] else 'FAIL':>5s}")
        L.append(f"  baseline $/question across the k reruns: "
                 + ", ".join(f"{x:.6f}" for x in b["usd_per_question_runs"]) + f" (sd {b['usd_per_question_sd']:.6f})")
        L.append("  saving = change in $/question vs the baseline mean; dF1/drec = lever minus baseline mean;")
        L.append("  gate PASS iff each gated drop <= its MDE (above).")
        L.append("")
        L.append("Paired bootstrap 95% CI of the drop (baseline per-question mean over k runs minus lever), context only")
        for name, s in S.items():
            if name == "baseline":
                continue
            parts = [f"{mm} {pb['mean_drop']:+.3f} [{pb['ci95'][0]:+.3f}, {pb['ci95'][1]:+.3f}]"
                     for mm, pb in s["paired_bootstrap"].items()]
            L.append(f"  {name:18s} " + "   ".join(parts))
        L.append("")

    L.append("Where the money goes (per question, per step; tokens from API usage)")
    L.append(f"  {'config/tag':22s} {'step':6s} {'model':18s} {'prompt':>7s} {'read%':>6s} {'out':>5s} {'$/q':>10s}")
    for r in R["runs"]:
        for step, d in r["agg"]["by_step"].items():
            L.append(f"  {r['config'] + '/' + r['tag']:22s} {step:6s} {d['model']:18s} {d['prompt_tokens_per_q']:7.0f} "
                     f"{100 * d['cache_read_share']:5.1f}% {d['output_tokens_per_q']:5.0f} {d['usd_per_q']:10.6f}")
    L.append("")
    L.append("Context bundle per run")
    for r in R["runs"]:
        a = r["agg"]
        L.append(f"  {r['config'] + '/' + r['tag']:22s} docs {a['bundle_docs_per_q']:5.1f}  est tokens "
                 f"{a['bundle_est_tokens_per_q']:6.0f}  dupes dropped {a['dropped_dupes_total']:3d}  "
                 f"truncated {a['truncated_total']:3d}  parse fails qgen {a['parse_failures']['qgen']} "
                 f"synth {a['parse_failures']['synth']}")
    L.append("")
    sp = R["spend"]
    if sp["projected_usd_if_real"] is not None:
        L.append(f"Projected spend if run for real: ${sp['projected_usd_if_real']:.4f} "
                 f"({sp['real_calls']} calls; reused {sp['reused_calls']}); cap ${sp['cap_usd']}")
    else:
        L.append(f"Spend: this invocation ${sp['this_invocation_usd']:.4f} ({sp['real_calls']} real calls, "
                 f"{sp['reused_calls']} reused from the response cache); experiment total (ledger) "
                 f"${sp['ledger_total_usd']:.4f}; cap ${sp['cap_usd']}")
    return "\n".join(L) + "\n"
