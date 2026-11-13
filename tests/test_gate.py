import math

import pytest
from scipy import stats as st

from costcut.gate import check, noise, paired_bootstrap, passes, sd


def test_sd_ddof1():
    assert sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_mde_formula_k3():
    runs = [0.60, 0.62, 0.64]  # sd = 0.02
    nz = noise("answer_f1", runs, n=30)
    se = 0.02 * math.sqrt(1 / 3 + 1)
    raw = (st.t.ppf(0.95, 2) + st.t.ppf(0.80, 2)) * se
    assert nz.sigma_run == pytest.approx(0.02)
    assert nz.se == pytest.approx(se)
    assert nz.mde_raw == pytest.approx(raw)
    assert nz.mde == pytest.approx(max(raw, 1 / 30))
    assert nz.mean == pytest.approx(0.62)


def test_mde_floor_when_noise_is_zero():
    nz = noise("bundle_recall", [0.9, 0.9, 0.9], n=30)
    assert nz.sigma_run == 0
    assert nz.mde == pytest.approx(1 / 30)


def test_needs_k_at_least_3():
    with pytest.raises(ValueError):
        noise("answer_f1", [0.5, 0.6], n=30)


def test_holds_at_exactly_mde_and_fails_just_above():
    nz = noise("bundle_recall", [0.9, 0.9, 0.9], n=20)  # mde = 0.05
    assert check(nz, [0.85]).holds
    assert not check(nz, [0.8499]).holds


def test_improvement_always_holds():
    nz = noise("answer_f1", [0.60, 0.62, 0.64], n=30)
    v = check(nz, [0.90])
    assert v.drop < 0 and v.holds


def test_pass_needs_every_metric():
    a = noise("answer_f1", [0.6, 0.6, 0.6], n=10)       # mde 0.1
    b = noise("bundle_recall", [0.9, 0.9, 0.9], n=10)   # mde 0.1
    assert passes([check(a, [0.55]), check(b, [0.85])])
    assert not passes([check(a, [0.55]), check(b, [0.7])])
    with pytest.raises(ValueError):
        passes([])


def test_paired_bootstrap_ci_contains_mean_and_is_deterministic():
    base = [1.0, 0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5]
    lev = [0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    r1 = paired_bootstrap(base, lev, iters=2000)
    r2 = paired_bootstrap(base, lev, iters=2000)
    assert r1 == r2
    assert r1["mean_drop"] == pytest.approx(0.1875)
    assert r1["ci95"][0] <= r1["mean_drop"] <= r1["ci95"][1]
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
