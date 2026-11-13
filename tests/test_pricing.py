import pytest

from costcut.pricing import BIG, SMALL, Usage, cost_usd, price_row


def test_plain_input_output_sonnet():
    # 1M input at $2 + 1M output at $10
    assert cost_usd(BIG, Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == pytest.approx(12.0)


def test_plain_haiku():
    assert cost_usd(SMALL, Usage(input_tokens=2_000, output_tokens=100)) == pytest.approx((2000 * 0.10 + 100 * 0.50) / 1e6)


def test_cache_read_and_write_priced_separately():
    u = Usage(input_tokens=100, output_tokens=50, cache_creation_input_tokens=1000, cache_read_input_tokens=4000)
    exp = (100 * 2.00 + 1000 * 2.50 + 4000 * 0.10 + 50 * 10.00) / 1e6
    assert cost_usd(BIG, u) == pytest.approx(exp)


def test_ttl_breakdown_1h_priced_at_1h_rate():
    u = Usage(cache_creation_input_tokens=1000, cache_write_5m=400, cache_write_1h=600)
    assert cost_usd(BIG, u) == pytest.approx((400 * 2.50 + 600 * 4.00) / 1e6)


def test_ttl_breakdown_must_sum():
    with pytest.raises(ValueError):
        cost_usd(BIG, Usage(cache_creation_input_tokens=1000, cache_write_5m=10))


def test_no_breakdown_means_5m():
    assert cost_usd(SMALL, Usage(cache_creation_input_tokens=1000)) == pytest.approx(1000 * 0.125 / 1e6)


def test_prompt_tokens_include_cache_fields():
    u = Usage(input_tokens=5, cache_creation_input_tokens=7, cache_read_input_tokens=11, output_tokens=100)
    assert u.prompt_tokens == 23


def test_haiku_long_context_tier_uses_total_prompt():
    # 60k uncached + 50k cache read = 110k prompt tokens > 100k -> long tier for ALL tokens
    u = Usage(input_tokens=60_000, cache_read_input_tokens=50_000, output_tokens=10)
    assert price_row(SMALL, u.prompt_tokens)["input"] == 0.50
    assert cost_usd(SMALL, u) == pytest.approx((60_000 * 0.50 + 50_000 * 0.05 + 10 * 2.50) / 1e6)
    assert price_row(SMALL, 100_000)["input"] == 0.10


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("claude-made-up", Usage(input_tokens=1))


def test_usage_add_and_roundtrip():
    a = Usage(input_tokens=1, output_tokens=2, cache_read_input_tokens=3)
    b = Usage(input_tokens=10, cache_creation_input_tokens=4, cache_write_5m=4)
    s = a + b
    assert s.to_dict() == {"input_tokens": 11, "output_tokens": 2, "cache_creation_input_tokens": 4,
                           "cache_read_input_tokens": 3, "cache_write_5m": 4, "cache_write_1h": 0}
    assert Usage.from_dict(s.to_dict()) == s
