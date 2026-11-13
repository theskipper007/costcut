import pytest

from costcut.metrics import em, f1, ndcg_at_k, parse_json, recall


def test_f1_basic_and_yes_no():
    assert f1("The Eiffel Tower", "eiffel tower") == 1.0
    assert f1("yes", "no") == 0.0
    assert f1("yes it is", "yes") == 0.0  # yes/no must match exactly
    assert f1("Paris France", "Paris") == pytest.approx(2 / 3)
    assert em("the  Paris.", "paris") == 1.0


def test_ndcg_perfect_and_duplicates_count_once():
    assert ndcg_at_k(["A", "B", "x"], ["A", "B"]) == pytest.approx(1.0)
    assert ndcg_at_k(["A", "A", "B"], ["A", "B"]) < 1.0
    assert ndcg_at_k(["x"] * 10 + ["A"], ["A"]) == 0.0


def test_recall():
    assert recall(["A", "x"], ["A", "B"]) == 0.5


def test_parse_json_lenient():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('here: {"a": 2} done') == {"a": 2}
    assert parse_json("nope") == {}
