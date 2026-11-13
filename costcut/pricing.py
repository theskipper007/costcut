"""Price table and the cost formula. Every dollar in results.json comes from here.

Prices: USD per 1M tokens, Anthropic first-party API, standard (non-batch, non-fast) tier.
Source: https://platform.claude.com/docs/en/about-claude/pricing (fetched 2026-11-13).

    cost = ( input_tokens                 * p_input
           + cache_write_5m_tokens        * p_cache_write_5m
           + cache_write_1h_tokens        * p_cache_write_1h
           + cache_read_input_tokens      * p_cache_read
           + output_tokens                * p_output ) / 1e6

Token counts are the API's own `usage` fields. On the Messages API `input_tokens` EXCLUDES
cached tokens: total prompt = input_tokens + cache_creation_input_tokens + cache_read_input_tokens.
Claude Haiku 5.5 has a long-context tier for prompts over 100,000 tokens; this pipeline never
gets near it, but the formula applies it anyway so a mistake cannot under-price.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

PRICES_DATE = "2026-11-13"
PRICES_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"

BIG = "claude-sonnet-5-5"
SMALL = "claude-haiku-5-5"

# model -> list of (max_prompt_tokens or None, {price fields})
PRICES: dict[str, list[tuple[int | None, dict[str, float]]]] = {
    BIG: [
        (None, {"input": 2.00, "cache_write_5m": 2.50, "cache_write_1h": 4.00,
                "cache_read": 0.10, "output": 10.00}),
    ],
    SMALL: [
        (100_000, {"input": 0.10, "cache_write_5m": 0.125, "cache_write_1h": 0.20,
                   "cache_read": 0.01, "output": 0.50}),
        (None, {"input": 0.50, "cache_write_5m": 0.625, "cache_write_1h": 1.00,
                "cache_read": 0.05, "output": 2.50}),
    ],
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(**{k: getattr(self, k) + getattr(o, k) for k in asdict(self)})

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Usage":
        return Usage(**{k: int(d.get(k, 0) or 0) for k in asdict(Usage())})


def price_row(model: str, prompt_tokens: int) -> dict[str, float]:
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to PRICES with its source date")
    for limit, row in PRICES[model]:
        if limit is None or prompt_tokens <= limit:
            return row
    raise AssertionError("unreachable: last tier has no limit")


def cost_usd(model: str, u: Usage) -> float:
    if u.cache_write_5m + u.cache_write_1h not in (0, u.cache_creation_input_tokens):
        raise ValueError("cache write TTL breakdown does not sum to cache_creation_input_tokens")
    # If the API gave no TTL breakdown, every write is the default 5-minute TTL.
    w5 = u.cache_write_5m if (u.cache_write_5m + u.cache_write_1h) else u.cache_creation_input_tokens
    p = price_row(model, u.prompt_tokens)
    return (u.input_tokens * p["input"] + w5 * p["cache_write_5m"] + u.cache_write_1h * p["cache_write_1h"]
            + u.cache_read_input_tokens * p["cache_read"] + u.output_tokens * p["output"]) / 1e6


def price_table() -> dict:
    return {"date": PRICES_DATE, "source": PRICES_SOURCE, "unit": "USD per 1M tokens",
            "models": {m: [{"max_prompt_tokens": lim, **row} for lim, row in tiers]
                       for m, tiers in PRICES.items()}}
