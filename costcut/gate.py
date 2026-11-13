"""The quality gate. Fixed before any lever ran (README, "Pre-registration", 2026-11-13).

For each GATED metric m (answer_f1, bundle_recall), with n questions:

    s_1..s_k   mean of m over the n questions, in each of k baseline reruns (k >= 3)
    x_1..x_r   the same mean for the lever's run(s)  (r = 1 here)
    sigma_run  sample SD of s_1..s_k (ddof = 1). RUN-TO-RUN noise: same questions, rerun.
    SE         sigma_run * sqrt(1/k + 1/r)             unpaired, lever assumed to have the
                                                       baseline's run-to-run variance
    MDE        max( (t_{1-alpha, k-1} + t_{power, k-1}) * SE ,  1/n )
               one-sided, alpha = 0.05, power = 0.80. The 1/n floor is the metric's resolution
               (one question moving by a whole point); it stops a lucky sigma of ~0 from making
               the gate demand a perfect match.
    drop       mean(s) - mean(x)                       positive = the lever is worse

    the lever HOLDS on m   iff  drop <= MDE
    the lever PASSES       iff  it holds on every gated metric

Read "holds" as "no drop bigger than this experiment can detect", not as "proven equivalent".
A lever can pass with a real drop smaller than the MDE. The report prints the MDE next to every
verdict so the reader can see how coarse the gate is, plus a paired bootstrap CI of the drop.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from scipy import stats as st

ALPHA = 0.05
EPS = 1e-12  # float tolerance only: 0.9 - 0.85 is 0.05000000000000004 in binary
POWER = 0.80
FORMULA = ("MDE = max((t_{1-alpha,k-1} + t_{power,k-1}) * sigma_run * sqrt(1/k + 1/r), 1/n); "
           "alpha=0.05 one-sided, power=0.80, sigma_run = SD(ddof=1) of the k baseline run means; "
           "lever holds on a metric iff (mean baseline - mean lever) <= MDE; passes iff it holds on "
           "every gated metric")


def sd(xs: list[float]) -> float:
    k = len(xs)
    if k < 2:
        raise ValueError("need k >= 2 baseline runs for a run-to-run SD")
    mu = sum(xs) / k
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (k - 1))


@dataclass
class Noise:
    metric: str
    n: int
    k: int
    r: int
    mean: float
    runs: list[float]
    sigma_run: float
    se: float
    mde_raw: float
    mde: float
    alpha: float = ALPHA
    power: float = POWER

    def to_dict(self) -> dict:
        return asdict(self)


def noise(metric: str, baseline_means: list[float], n: int, r: int = 1,
          alpha: float = ALPHA, power: float = POWER) -> Noise:
    k = len(baseline_means)
    if k < 3:
        raise ValueError("the gate needs k >= 3 baseline reruns")
    s = sd(baseline_means)
    se = s * math.sqrt(1 / k + 1 / r)
    raw = float((st.t.ppf(1 - alpha, k - 1) + st.t.ppf(power, k - 1)) * se)
    return Noise(metric=metric, n=n, k=k, r=r, mean=sum(baseline_means) / k, runs=list(baseline_means),
                 sigma_run=s, se=se, mde_raw=raw, mde=max(raw, 1 / n), alpha=alpha, power=power)


@dataclass
class Verdict:
    metric: str
    baseline: float
    lever: float
    drop: float
    mde: float
    holds: bool


def check(nz: Noise, lever_means: list[float]) -> Verdict:
    lv = sum(lever_means) / len(lever_means)
    drop = nz.mean - lv
    return Verdict(metric=nz.metric, baseline=nz.mean, lever=lv, drop=drop, mde=nz.mde,
                   holds=bool(drop <= nz.mde + EPS))


def passes(verdicts: list[Verdict]) -> bool:
    if not verdicts:
        raise ValueError("no gated metrics")
    return all(v.holds for v in verdicts)


def paired_bootstrap(base_per_q: list[float], lever_per_q: list[float], iters: int = 10_000,
                     seed: int = 0) -> dict:
    """95% percentile CI of mean(base - lever) over questions. base_per_q is each question's
    mean over the k baseline runs. Reported for context; the gate does not use it."""
    if len(base_per_q) != len(lever_per_q) or not base_per_q:
        raise ValueError("need equal, non-empty per-question lists")
    d = [b - x for b, x in zip(base_per_q, lever_per_q)]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"mean_drop": sum(d) / n, "ci95": [means[int(0.025 * iters)], means[int(0.975 * iters) - 1]],
            "iters": iters, "seed": seed}
