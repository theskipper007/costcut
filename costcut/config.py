"""Pipeline configurations: the baseline, each lever on its own, and all three combined.
Fixed before any real call (README, "Pre-registration", 2026-11-13). Not tuned afterwards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .pricing import BIG, SMALL

# Context-budget sweep (estimated tokens, ceil(chars/4)). The generous baseline bundle is the
# 20 fused candidates, whole, which is ~2,900 estimated tokens on this corpus (mean paragraph
# ~142). The sweep cuts that to roughly 2/3, 1/3 and 1/6.
BUDGETS = (2000, 1000, 500)
PER_SOURCE_CAP = 200
COMBINED_BUDGET = 1000  # the middle budget, chosen up front, not after seeing the sweep

MAX_TOKENS = {"intent": 64, "qgen": 300, "synth": 400}


@dataclass(frozen=True)
class Config:
    name: str
    lever: str
    intent_model: str = BIG
    qgen_model: str = BIG
    synth_model: str = BIG
    budget: int | None = None          # None = generous: all candidates, whole
    per_source_cap: int | None = None
    dedup: bool = False
    caching: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE = Config("baseline", "none", description="every step on the larger tier, all 20 candidates "
                  "whole (~2.9k est. tokens), no prompt caching")


def levers() -> list[Config]:
    out = [replace(BASELINE, name=f"context_{b}", lever="context", budget=b, per_source_cap=PER_SOURCE_CAP,
                   dedup=True, description=f"hard {b}-token budget on the bundle, {PER_SOURCE_CAP}-token "
                   "per-source cap, content-hash dedup") for b in BUDGETS]
    out.append(replace(BASELINE, name="routing_upstream", lever="routing", intent_model=SMALL, qgen_model=SMALL,
                       description="intent + query generation on the cheap tier; synthesis stays large"))
    out.append(replace(BASELINE, name="routing_all", lever="routing", intent_model=SMALL, qgen_model=SMALL,
                       synth_model=SMALL, description="every step on the cheap tier, synthesis included"))
    out.append(replace(BASELINE, name="caching", lever="caching", caching=True,
                       description="cache_control breakpoint at the end of each step's stable system prefix"))
    out.append(replace(BASELINE, name="combined", lever="combined", intent_model=SMALL, qgen_model=SMALL,
                       budget=COMBINED_BUDGET, per_source_cap=PER_SOURCE_CAP, dedup=True, caching=True,
                       description=f"routing_upstream + context_{COMBINED_BUDGET} + caching"))
    return out


ALL = [BASELINE] + levers()
BY_NAME = {c.name: c for c in ALL}
