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
The evaluation functions live in :mod:`src.evaluation` and are shared by
all detectors.

Run::

    python -m src.threshold_baseline
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass

import pandas as pd

from src import config
from src.data_loading import load_metropt3
from src.evaluation import _run_ids, evaluate_events, warning_window_labels

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
