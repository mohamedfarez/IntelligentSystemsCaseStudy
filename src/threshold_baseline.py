"""Rule-based threshold baseline for air-leak early warning on MetroPT-3.

Idea
----
An air leak forces the compressor to work harder to keep the system
pressurised. We therefore aggregate the raw signals into fixed windows and
check four simple rules:

* ``tp3_min``            unusually LOW  (pneumatic panel pressure drops)
* ``reservoirs_min``     unusually LOW  (reservoir pressure drops)
* ``load_ratio``         unusually HIGH (share of time the motor runs under load)
* ``motor_current_mean`` unusually HIGH

"Unusual" thresholds are quantiles estimated **only on the chronological
training period** (first month, normal operation), so no information from
the test period leaks into the baseline. A window raises an alarm when at
least ``min_rules`` rules fire for ``persistence`` consecutive windows;
windows separated by a recording gap are not considered consecutive.

Timing convention
-----------------
Every window is indexed by its **end** time, i.e. the moment at which all
its data is available and a decision can be made. Lead times are therefore
not inflated by the window length.

Evaluation is event-based: a failure counts as detected if an alarm occurs
between ``start - warning_horizon`` and the end of the failure; lead time
(minutes) is measured from the first such alarm to the reported failure
start (positive = before the failure, negative = detection delay). Because
only alarms inside the horizon are counted, lead time cannot exceed the
horizon. Alarm episodes outside these windows are counted as false alarms.
Window-level precision/recall/F1 are reported as a secondary view.

Run::

    python -m src.threshold_baseline
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src import config
from src.data_loading import load_metropt3

logger = logging.getLogger(__name__)

#: (feature name, direction) — direction "low" fires below the lower
#: quantile threshold, "high" fires above the upper quantile threshold.
RULES: list[tuple[str, str]] = [
    ("tp3_min", "low"),
    ("reservoirs_min", "low"),
    ("load_ratio", "high"),
    ("motor_current_mean", "high"),
]


@dataclass
class BaselineConfig:
    """Hyper-parameters of the threshold baseline.

    Attributes:
        window: Aggregation window (pandas offset alias).
        low_quantile: Training quantile used for "low" rules.
        high_quantile: Training quantile used for "high" rules.
        min_rules: Number of rules that must fire in a window.
        persistence: Consecutive alarming windows required to raise an alarm.
        warning_horizon: How long before a failure an alarm counts as an
            early warning (provisional choice for this milestone).
        load_current_a: Motor current above which the compressor is
            considered to run under load.
    """

    window: str = "10min"
    low_quantile: float = 0.01
    high_quantile: float = 0.99
    min_rules: int = 2
    persistence: int = 3
    warning_horizon: str = "2h"
    load_current_a: float = config.MOTOR_LOAD_CURRENT_A


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def window_features(df: pd.DataFrame, window: str = "10min",
                    load_current_a: float = config.MOTOR_LOAD_CURRENT_A) -> pd.DataFrame:
    """Aggregate raw signals into fixed windows indexed by their END time.

    Windows are left-closed ``[t - window, t)``; the index ``t`` is the
    earliest time at which the window's features are known. Windows
    without any raw sample (recording gaps) are dropped.

    Args:
        df: Time-indexed frame from :func:`src.data_loading.load_metropt3`.
        window: Pandas offset alias, e.g. ``"10min"``.
        load_current_a: Threshold separating offloaded and loaded motor states.

    Returns:
        Data frame indexed by window end time (``window_end``) with the
        features used by :data:`RULES`.
    """
    under_load = (df["Motor_current"] > load_current_a).astype("float32")
    feats = pd.DataFrame({
        "tp3_min": df["TP3"].resample(window).min(),
        "reservoirs_min": df["Reservoirs"].resample(window).min(),
        "load_ratio": under_load.resample(window).mean(),
        "motor_current_mean": df["Motor_current"].resample(window).mean(),
    }).dropna()
    feats.index = feats.index + pd.Timedelta(window)
    feats.index.name = "window_end"
    return feats


def _run_ids(flags: pd.Series, step: pd.Timedelta) -> pd.Series:
    """Label runs of True values that are consecutive in *time*.

    A new run starts at every False value and wherever the time difference
    to the previous window is not exactly ``step`` (a recording gap).
    """
    t = pd.Series(flags.index, index=flags.index)
    breaks = (~flags) | (t.diff() != step)
    return breaks.cumsum()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ThresholdBaseline:
    """Quantile-threshold detector with a rule-count and persistence filter."""

    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        """Create an unfitted baseline.

        Args:
            cfg: Hyper-parameters; defaults to :class:`BaselineConfig`.
        """
        self.cfg = cfg or BaselineConfig()
        self.thresholds_: dict[str, float] = {}

    def fit(self, train_features: pd.DataFrame) -> "ThresholdBaseline":
        """Estimate one threshold per rule from normal training windows.

        Args:
            train_features: Window features from the training period only.

        Returns:
            The fitted instance.
        """
        for feat, direction in RULES:
            q = self.cfg.low_quantile if direction == "low" else self.cfg.high_quantile
            self.thresholds_[feat] = float(train_features[feat].quantile(q))
        return self

    def rule_violations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return a boolean frame with one column per rule.

        Args:
            features: Window features.

        Returns:
            Boolean data frame (True = rule fired in that window).
        """
        if not self.thresholds_:
            raise RuntimeError("Call fit() before rule_violations().")
        out = {}
        for feat, direction in RULES:
            thr = self.thresholds_[feat]
            out[feat] = features[feat] < thr if direction == "low" else features[feat] > thr
        return pd.DataFrame(out, index=features.index)

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Number of rules fired per window (0 … len(RULES))."""
        return self.rule_violations(features).sum(axis=1).rename("n_rules_fired")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Final alarm per window after rule-count and persistence filters.

        An alarm is raised in a window when it and the preceding
        ``persistence - 1`` windows all satisfy the rule-count condition and
        are contiguous in time (no recording gap in between).

        Args:
            features: Window features indexed by window end time.

        Returns:
            Boolean series (True = alarm).
        """
        raw = self.score(features) >= self.cfg.min_rules
        run_id = _run_ids(raw, pd.Timedelta(self.cfg.window))
        run_length = raw.astype(int).groupby(run_id.to_numpy()).cumsum()
        return (run_length >= self.cfg.persistence).rename("alarm")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def warning_window_labels(index: pd.DatetimeIndex, horizon: str,
                          periods: pd.DataFrame = config.FAILURE_PERIODS) -> pd.Series:
    """Label windows inside ``[failure start - horizon, failure end]``.

    Args:
        index: Window (end) times.
        horizon: Early-warning horizon (pandas Timedelta string).
        periods: Failure periods with ``start`` and ``end`` columns.

    Returns:
        Boolean series (True = window should raise an alarm).
    """
    h = pd.Timedelta(horizon)
    mask = np.zeros(len(index), dtype=bool)
    for start, end in zip(periods["start"], periods["end"]):
        mask |= (index >= start - h) & (index <= end)
    return pd.Series(mask, index=index, name="target")


def alarm_episodes(alarms: pd.Series, window: str) -> pd.DataFrame:
    """Merge alarming windows that are consecutive in time into episodes.

    Args:
        alarms: Boolean alarm series indexed by window end time.
        window: Window length, used to detect recording gaps.

    Returns:
        Data frame with ``start`` and ``end`` (first / last alarm time) per episode.
    """
    a = alarms.astype(bool)
    run_id = _run_ids(a, pd.Timedelta(window))[a]
    if run_id.empty:
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"),
                             "end": pd.Series(dtype="datetime64[ns]")})
    grouped = run_id.index.to_series().groupby(run_id.to_numpy())
    return pd.DataFrame({"start": grouped.min().to_numpy(), "end": grouped.max().to_numpy()})


def evaluate_events(alarms: pd.Series, horizon: str, window: str,
                    periods: pd.DataFrame = config.FAILURE_PERIODS) -> tuple[pd.DataFrame, dict]:
    """Event-level evaluation: detection, lead time and false alarms.

    Args:
        alarms: Boolean alarm series (test period only), indexed by window end.
        horizon: Early-warning horizon.
        window: Window length (for episode merging).
        periods: Failure periods to evaluate against.

    Returns:
        ``(events, summary)`` — one row per failure, plus a dict of KPIs.
        ``lead_time_min`` > 0 means the alarm came before the reported
        failure start; < 0 means it came after (detection delay).
    """
    h = pd.Timedelta(horizon)
    alarm_times = alarms.index[alarms.to_numpy(dtype=bool)]
    in_test = periods[(periods["end"] >= alarms.index.min()) & (periods["start"] <= alarms.index.max())]

    rows = []
    for _, p in in_test.iterrows():
        hits = alarm_times[(alarm_times >= p["start"] - h) & (alarm_times <= p["end"])]
        first = hits.min() if len(hits) else pd.NaT
        rows.append({
            "failure_id": p["failure_id"],
            "start": p["start"],
            "end": p["end"],
            "detected": len(hits) > 0,
            "first_alarm": first,
            "lead_time_min": (p["start"] - first) / pd.Timedelta("1min") if len(hits) else np.nan,
            "early_warning": bool(len(hits) and first < p["start"]),
        })
    events = pd.DataFrame(rows)

    episodes = alarm_episodes(alarms, window)
    is_true = np.zeros(len(episodes), dtype=bool)
    for _, p in in_test.iterrows():
        is_true |= ((episodes["end"] >= p["start"] - h) & (episodes["start"] <= p["end"])).to_numpy()
    n_false = int((~is_true).sum())
    test_days = (alarms.index.max() - alarms.index.min()) / pd.Timedelta("1D")

    y_true = warning_window_labels(alarms.index, horizon, in_test).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, alarms.astype(int), average="binary", zero_division=0
    )
    has_events = len(events) > 0
    summary = {
        "n_failures": len(events),
        "n_detected": int(events["detected"].sum()) if has_events else 0,
        "n_early_warnings": int(events["early_warning"].sum()) if has_events else 0,
        "event_recall": float(events["detected"].mean()) if has_events else float("nan"),
        "median_lead_time_min": float(events["lead_time_min"].median()) if has_events else float("nan"),
        "n_alarm_episodes": len(episodes),
        "n_false_alarm_episodes": n_false,
        "false_alarms_per_day": n_false / test_days if test_days > 0 else float("nan"),
        "window_precision": float(prec),
        "window_recall": float(rec),
        "window_f1": float(f1),
    }
    return events, summary


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------
def run_baseline(df: pd.DataFrame, cfg: BaselineConfig | None = None,
                 train_end: pd.Timestamp = config.TRAIN_END) -> dict:
    """Fit on the training period, predict and evaluate on the test period.

    A window belongs to the training period if it ends at or before
    ``train_end``. Training windows that overlap a failure warning window are
    removed so the thresholds describe normal behaviour only.

    Args:
        df: Raw time-indexed frame.
        cfg: Baseline hyper-parameters.
        train_end: Boundary between training and test period.

    Returns:
        Dict with ``model``, ``features``, ``alarms``, ``events`` and ``summary``.
    """
    cfg = cfg or BaselineConfig()
    feats = window_features(df, cfg.window, cfg.load_current_a)
    train = feats[feats.index <= train_end]
    train = train[~warning_window_labels(train.index, cfg.warning_horizon)]
    test = feats[feats.index > train_end]
    if train.empty or test.empty:
        raise ValueError("Empty training or test period - check TRAIN_END and the data range.")

    model = ThresholdBaseline(cfg).fit(train)
    alarms = model.predict(test)
    events, summary = evaluate_events(alarms, cfg.warning_horizon, cfg.window)
    logger.info("Train windows: %d | test windows: %d", len(train), len(test))
    return {"model": model, "features": feats, "alarms": alarms,
            "events": events, "summary": summary}


def main() -> None:
    """Command-line entry point: run the baseline and save result tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    defaults = BaselineConfig()
    parser.add_argument("--window", default=defaults.window)
    parser.add_argument("--min-rules", type=int, default=defaults.min_rules)
    parser.add_argument("--persistence", type=int, default=defaults.persistence)
    parser.add_argument("--horizon", default=defaults.warning_horizon)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.set_global_seed()
    cfg = BaselineConfig(window=args.window, min_rules=args.min_rules,
                         persistence=args.persistence, warning_horizon=args.horizon)
    out = run_baseline(load_metropt3(), cfg)

    print("Config:", asdict(cfg))
    print("Thresholds:", {k: round(v, 3) for k, v in out["model"].thresholds_.items()})
    print("\nPer-failure results:\n", out["events"].to_string(index=False))
    print("\nSummary:")
    for k, v in out["summary"].items():
        print(f"  {k:24s} {v}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out["events"].to_csv(config.RESULTS_DIR / "baseline_threshold_events.csv", index=False)
    pd.Series(out["summary"]).to_csv(config.RESULTS_DIR / "baseline_threshold_summary.csv",
                                     header=["value"])
    print(f"\nSaved tables to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
