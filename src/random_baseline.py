"""Random-alarm baseline: what detection and lead time occur by chance?

With only four failures and alarms that cover a noticeable share of the test
period, some failures can be "detected" by chance. Following the concern of
Kim et al. (2022) that random scores can look good under weak evaluation
protocols, this baseline answers: if alarms were raised at random, how
often would the event-level metrics look as good as a real detector's?

Construction (repeated for ``n_runs`` seeds):

1. every test window exceeds the threshold independently with probability
   ``exceed_rate`` (set to the exceedance rate of the detector being
   compared, so both raise a similar amount of alarms);
2. the exceedances pass through the same low-pass filter as the learned
   detectors (:mod:`src.alarm_logic`);
3. the resulting alarms are evaluated with :func:`src.evaluation.evaluate_events`.

The baseline uses the test timestamps only to place its random alarms; it
learns nothing from the test data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import config
from src.alarm_logic import AlarmConfig, low_pass
from src.evaluation import evaluate_events


def random_alarms(index: pd.DatetimeIndex, exceed_rate: float, rng: np.random.Generator,
                  alarm_cfg: AlarmConfig | None = None) -> pd.Series:
    """Random exceedances passed through the common low-pass filter."""
    cfg = alarm_cfg or AlarmConfig()
    exceed = pd.Series(rng.random(len(index)) < exceed_rate, index=index)
    return (low_pass(exceed, cfg.alpha, cfg.window) > cfg.level).rename("alarm")


def run_random_baseline(index: pd.DatetimeIndex, exceed_rate: float, horizon: str = "2h",
                        n_runs: int = 100, seed: int = config.RANDOM_SEED,
                        alarm_cfg: AlarmConfig | None = None) -> pd.DataFrame:
    """Evaluate ``n_runs`` random-alarm runs.

    Args:
        index: Test window end times (same windows as the compared detector).
        exceed_rate: Probability that a window exceeds the threshold.
        horizon: Early-warning horizon.
        n_runs: Number of random repetitions.
        seed: Seed of the random generator (runs are reproducible).
        alarm_cfg: Alarm-logic parameters (filter, window length).

    Returns:
        One row per run with the KPI summary of :func:`evaluate_events`.
    """
    cfg = alarm_cfg or AlarmConfig()
    rng = np.random.default_rng(seed)
    rows = []
    with warnings.catch_warnings():
        # runs without any detected failure have an undefined (NaN) median lead time
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_runs):
            alarms = random_alarms(index, exceed_rate, rng, cfg)
            _, summary = evaluate_events(alarms, horizon, cfg.window)
            rows.append(summary)
    return pd.DataFrame(rows)
