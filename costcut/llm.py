"""One LLM call path: on-disk response cache, a spend ledger, a hard USD cap, and backends.

Response cache: keyed by sha256 of the full request (model, system blocks incl. cache_control,
messages, max_tokens, thinking) PLUS a replicate tag. Same request + same tag -> the stored
response is reused (no API call, no spend). Different tag -> a fresh, independent sample.
The cached file keeps the API's usage fields, so the cost of a reused response is the cost
it really had when it was made.

Spend ledger (runs/ledger.jsonl, committed): one line per REAL API call with its usage and
cost. Total experiment spend = sum of the ledger, across every invocation, aborted ones too.
The --max-usd cap is checked against the ledger total before every real call.

Backends:
    anthropic  Claude Messages API (official SDK). Thinking is turned off on both tiers so
               the cost comparison is about the levers, not about hidden reasoning tokens.
    fake       offline stub for --dry-run. Its answers are junk; it exists to exercise the
               plumbing and to project spend (chars/4 x 1.3 tokens, every token priced as
               uncached input, so the projection errs high).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .pricing import Usage, cost_usd

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "runs" / "cache"
FAKE_CACHE_DIR = ROOT / ".cache" / "fake"
LEDGER = ROOT / "runs" / "ledger.jsonl"

FAKE_TOKEN_FUDGE = 1.3
FAKE_OUTPUT_TOKENS = {"intent": 12, "qgen": 80, "synth": 120}

# Per-tier "thinking off" settings, as the API accepts them (claude-api docs, 2026-11):
# Claude Sonnet 5.5 rejects {"type": "disabled"}; its off switch is {"type": "between_tools"}.
# Claude Haiku 5.5 accepts {"type": "disabled"} at effort high or below (default is medium).
THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"},
                "claude-haiku-5-5": {"type": "disabled"}}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Response:
    text: str
    usage: Usage
    model: str
    stop_reason: str
    cost: float
    reused: bool = False  # served from the response cache (no new spend)


@dataclass
class Spend:
    real_calls: int = 0
    reused_calls: int = 0
    usd: float = 0.0  # real spend in THIS invocation
    by_model: dict = field(default_factory=dict)


def ledger_total(path: Path = LEDGER) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


class LLM:
    def __init__(self, backend: str, max_usd: float | None, cache: bool = True):
        self.backend = backend
        self.max_usd = max_usd
        self.cache = cache
        self.cache_dir = FAKE_CACHE_DIR if backend == "fake" else CACHE_DIR
        if backend == "fake" and FAKE_CACHE_DIR.exists():
            import shutil
            shutil.rmtree(FAKE_CACHE_DIR)  # each dry run projects from scratch
        self.spend = Spend()
        self._lock = threading.Lock()
        self._ledger_usd = ledger_total() if backend == "anthropic" else 0.0
        self._client = None
        if backend == "anthropic":
            import anthropic
            # Pin the endpoint so an inherited ANTHROPIC_BASE_URL cannot redirect the run.
            # The key is read from the environment by name and never printed or logged.
            self._client = anthropic.Anthropic(
                base_url=os.environ.get("COSTCUT_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=0)

    @property
    def ledger_usd(self) -> float:
        return self._ledger_usd

    def call(self, *, step: str, model: str, system_blocks: list[dict], user: str,
             max_tokens: int, tag: str) -> Response:
        req = {"model": model, "system": system_blocks, "messages": [{"role": "user", "content": user}],
               "max_tokens": max_tokens, "thinking": THINKING_OFF.get(model), "tag": tag}
        key = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if self.cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            u = Usage.from_dict(d["usage"])
            r = Response(text=d["text"], usage=u, model=d["model"], stop_reason=d["stop_reason"],
                         cost=cost_usd(model, u), reused=True)
            with self._lock:
                self.spend.reused_calls += 1
            return r
        with self._lock:
            if self.max_usd is not None and self._ledger_usd >= self.max_usd:
                raise BudgetExceeded(f"ledger spend ${self._ledger_usd:.4f} reached the ${self.max_usd} cap")
        r = getattr(self, f"_{self.backend}")(req, step)
        with self._lock:
            self.spend.real_calls += 1
            self.spend.usd += r.cost if self.backend == "anthropic" else 0.0
            bm = self.spend.by_model.setdefault(model, {"calls": 0, "usd": 0.0})
            bm["calls"] += 1
            bm["usd"] += r.cost
            if self.backend == "anthropic":
                self._ledger_usd += r.cost
                LEDGER.parent.mkdir(parents=True, exist_ok=True)
                with LEDGER.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": key, "step": step,
                                        "model": model, "tag": tag, "usage": r.usage.to_dict(),
                                        "usd": r.cost}) + "\n")
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": r.text, "usage": r.usage.to_dict(), "model": r.model,
                                        "stop_reason": r.stop_reason, "step": step, "tag": tag},
                                       ensure_ascii=False), encoding="utf-8")
        return r

    # ---- backends -----------------------------------------------------------
    def _anthropic(self, req: dict, step: str) -> Response:  # pragma: no cover - needs credentials
        import anthropic
        kw = dict(model=req["model"], max_tokens=req["max_tokens"], system=req["system"],
                  messages=req["messages"])
        if req["thinking"]:
            kw["thinking"] = req["thinking"]
        for attempt in range(7):
            try:
                m = self._client.messages.create(**kw)
                break
            except (anthropic.RateLimitError, anthropic.APIConnectionError,
                    anthropic.InternalServerError, anthropic.APITimeoutError):
                if attempt == 6:
                    raise
                time.sleep(min(60, 2 ** attempt))
        u = m.usage
        cc = getattr(u, "cache_creation", None)
        usage = Usage(input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                      cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
                      cache_read_input_tokens=u.cache_read_input_tokens or 0,
                      cache_write_5m=(getattr(cc, "ephemeral_5m_input_tokens", 0) or 0) if cc else 0,
                      cache_write_1h=(getattr(cc, "ephemeral_1h_input_tokens", 0) or 0) if cc else 0)
        text = "".join(b.text for b in m.content if b.type == "text")
        return Response(text=text, usage=usage, model=m.model, stop_reason=m.stop_reason or "",
                        cost=cost_usd(req["model"], usage))

    def _fake(self, req: dict, step: str) -> Response:
        sys_text = "".join(b["text"] for b in req["system"])
        user = req["messages"][0]["content"]
        n_in = int(FAKE_TOKEN_FUDGE * (len(sys_text) + len(user)) / 4)
        n_out = int(FAKE_TOKEN_FUDGE * FAKE_OUTPUT_TOKENS[step])
        q = re.search(r"Question: (.*)", user)
        question = q.group(1) if q else user
        if step == "intent":
            text = json.dumps({"intent": "comparison" if question.lower().startswith(("were", "are", "is", "which")) else "bridge"})
        elif step == "qgen":
            text = json.dumps({"queries": [question]})
        else:
            m = re.search(r"\[1\] ([^:\n]+):", user)
            text = json.dumps({"answer": m.group(1) if m else "unknown", "citations": [1]})
        u = Usage(input_tokens=n_in, output_tokens=n_out)
        return Response(text=text, usage=u, model="fake", stop_reason="end_turn",
                        cost=cost_usd(req["model"], u))
