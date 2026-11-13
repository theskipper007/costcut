"""costcut CLI.

    python -m costcut run --dry-run     offline fake backend: plumbing check + projected spend
    python -m costcut run               the real run (Anthropic API; needs ANTHROPIC_API_KEY)
    python -m costcut report            re-render report.txt from results.json
    python -m costcut smoke             one tiny real call per tier (checks IDs and thinking-off)

A rerun of `python -m costcut run` with the committed runs/cache reuses every stored response
and reproduces results.json exactly at $0. Add --no-cache for a fresh, independent run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_N = 30
DEFAULT_K = 3
DEFAULT_MAX_USD = 3.50  # hard cap on the ledger total (raised from 2.80 with Arun's OK, 2026-11-13)


def load_key(path: Path) -> None:
    """Set ANTHROPIC_API_KEY from a KEY=VALUE file, unless the environment already has it.
    Reads only that one key. The value is never printed, logged, cached or written anywhere."""
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == "ANTHROPIC_API_KEY":
            os.environ.setdefault("ANTHROPIC_API_KEY", v.strip().strip('"').strip("'"))
            return
    raise SystemExit(f"no ANTHROPIC_API_KEY line in {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="costcut")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true", help="fake backend, no API calls, no real numbers")
    r.add_argument("-n", type=int, default=DEFAULT_N, help=f"questions, rows 0..n-1 (default {DEFAULT_N})")
    r.add_argument("-k", type=int, default=DEFAULT_K, help=f"baseline reruns (default {DEFAULT_K}, min 3)")
    r.add_argument("--only", default="", help="comma-separated lever configs (default: all)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD, help="hard cap on total experiment spend")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--out", type=Path, default=None)
    sub.add_parser("report")
    sm = sub.add_parser("smoke")
    tk = sub.add_parser("tokens", help="count each stable prefix with the free count_tokens endpoint")
    for p_ in (r, sm, tk):
        p_.add_argument("--env-file", type=Path, default=None,
                        help="read ANTHROPIC_API_KEY (only that key) from this KEY=VALUE file into the "
                             "process environment; nothing is copied, printed or logged")
    a = ap.parse_args(argv)
    if getattr(a, "env_file", None):
        load_key(a.env_file)

    if a.cmd == "report":
        from .report import render
        res = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
        (ROOT / "report.txt").write_text(render(res), encoding="utf-8")
        print(render(res))
        return 0

    if a.cmd in ("smoke", "tokens") or (a.cmd == "run" and not a.dry_run):
        if not (os.environ.get("ANTHROPIC_API_KEY")):
            ap.error("ANTHROPIC_API_KEY is not set in the environment (or use --dry-run)")

    if a.cmd == "smoke":
        from .llm import LLM
        from .pricing import BIG, SMALL
        llm = LLM("anthropic", max_usd=DEFAULT_MAX_USD, cache=False)
        for model in (SMALL, BIG):
            resp = llm.call(step="intent", model=model, system_blocks=[{"type": "text", "text": "Reply with OK."}],
                            user="ping", max_tokens=8, tag="smoke")
            print(model, "->", resp.model, repr(resp.text), resp.usage.to_dict(), f"${resp.cost:.6f}")
        return 0

    if a.cmd == "tokens":
        import anthropic
        from . import prompts as P
        from .pricing import BIG, SMALL
        c = anthropic.Anthropic(base_url="https://api.anthropic.com", api_key=os.environ["ANTHROPIC_API_KEY"])
        for model in (BIG, SMALL):
            for name in ("INTENT_SYSTEM", "QGEN_SYSTEM", "SYNTH_SYSTEM"):
                t = c.messages.count_tokens(model=model, system=getattr(P, name),
                                            messages=[{"role": "user", "content": "x"}]).input_tokens
                print(f"{model:18s} {name:14s} {t} tokens (system + 1-char user message)")
        return 0

    if a.k < 3:
        ap.error("-k must be >= 3 (the gate needs a run-to-run SD)")
    from .runner import run
    only = [x.strip() for x in a.only.split(",") if x.strip()] or None
    backend = "fake" if a.dry_run else "anthropic"
    run(backend=backend, n=a.n, k=a.k, workers=a.workers, max_usd=a.max_usd if backend == "anthropic" else None,
        only=only, out=a.out, cache=not a.no_cache)
    return 0
