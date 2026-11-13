from costcut.context import build_bundle, cap_text, est_tokens
from costcut.data import Doc


def docs():
    return [Doc(0, "A", "a" * 400), Doc(1, "B", "b" * 400), Doc(2, "A", "a" * 400), Doc(3, "C", "c" * 40)]


def test_generous_bundle_keeps_everything_in_rank_order():
    b = build_bundle(docs(), [3, 0, 1, 2], budget=None, per_source_cap=None, dedup=False)
    assert [d[0] for d in b.docs] == [3, 0, 1, 2]
    assert b.dropped_dupes == 0 and not b.stopped_at_budget


def test_dedup_drops_identical_content():
    b = build_bundle(docs(), [0, 2, 1], budget=None, per_source_cap=None, dedup=True)
    assert [d[0] for d in b.docs] == [0, 1]
    assert b.dropped_dupes == 1


def test_hard_budget_stops_and_never_exceeds():
    b = build_bundle(docs(), [0, 1, 3], budget=150, per_source_cap=None, dedup=True)
    assert [d[0] for d in b.docs] == [0]  # A (~101) fits, B would overflow -> stop
    assert b.stopped_at_budget
    assert b.est_tokens <= 150


def test_cap_text_cuts_at_sentence_boundary():
    t = "One two three. " * 20
    out = cap_text(t.strip(), 10)
    assert out.endswith(".")
    assert est_tokens(out) <= 10


def test_cap_text_hard_truncates_a_single_long_sentence():
    out = cap_text("x" * 1000, 50)
    assert len(out) == 200


def test_per_source_cap_counts_truncations():
    b = build_bundle(docs(), [0, 3], budget=None, per_source_cap=20, dedup=False)
    assert b.truncated == 1
