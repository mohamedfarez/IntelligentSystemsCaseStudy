"""End-to-end experiment runner for the learned detectors.

Pipeline: preprocessing (frozen-segment masking, gap-aware windows,
chronological split, train-only scaling) -> detector fit on February ->
anomaly scores -> common alarm logic -> event-level evaluation, including
lead time relative to the first LPS activation.

Run::

    python -m src.run_experiment --detector iforest
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

import pandas as pd

from src import config
from src.alarm_logic import AlarmConfig, AlarmRule
from src.data_loading import load_metropt3
from src.detectors import IsolationForestDetector
from src.evaluation import evaluate_events, lps_onsets
from src.preprocessing import PreprocessConfig, prepare

logger = logging.getLogger(__name__)

DETECTORS = {"iforest": IsolationForestDetector}


def run(df: pd.DataFrame, detector_name: str = "iforest",
        pre_cfg: PreprocessConfig | None = None,
        alarm_cfg: AlarmConfig | None = None,
        horizon: str = "2h") -> dict:
    """Run one detector through the full pipeline.

    Args:
        df: Raw time-indexed frame.
        detector_name: Key in :data:`DETECTORS`.
        pre_cfg: Preprocessing parameters.
        alarm_cfg: Alarm-logic parameters.
        horizon: Early-warning horizon for the evaluation.

    Returns:
        Dict with the prepared data, detector, scores, alarm rule, alarms,
        per-failure events and the KPI summary.
    """
    config.set_global_seed()
    pre_cfg = pre_cfg or PreprocessConfig()
    data = prepare(df, pre_cfg)
    logger.info("Removed %d frozen records; train windows %d, test windows %d",
                data.n_records_removed, len(data.train), len(data.test))

    detector = DETECTORS[detector_name]().fit(data.train)
    train_scores = detector.score(data.train)
    test_scores = detector.score(data.test)

    alarm_cfg = alarm_cfg or AlarmConfig(window=pre_cfg.window)
    rule = AlarmRule(alarm_cfg).fit(train_scores)
    alarms = rule.predict(test_scores)
    events, summary = evaluate_events(alarms, horizon, pre_cfg.window,
                                      lps=lps_onsets(df, horizon))
    return {"data": data, "detector": detector, "train_scores": train_scores,
            "test_scores": test_scores, "rule": rule, "alarms": alarms,
            "events": events, "summary": summary}


def main() -> None:
    """Command-line entry point: run a detector and save result tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detector", default="iforest", choices=sorted(DETECTORS))
    parser.add_argument("--no-frozen-mask", action="store_true",
                        help="keep frozen segments (for the ablation)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pre_cfg = PreprocessConfig(mask_frozen=not args.no_frozen_mask)
    out = run(load_metropt3(), args.detector, pre_cfg)

    print("Preprocessing:", asdict(pre_cfg))
    print("Alarm logic:", asdict(out["rule"].cfg), "| threshold:", round(out["rule"].threshold_, 4))
    print("\nPer-failure results:\n", out["events"].to_string(index=False))
    print("\nSummary:")
    for k, v in out["summary"].items():
        print(f"  {k:24s} {v}")

    tag = args.detector + ("_nomask" if args.no_frozen_mask else "")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out["events"].to_csv(config.RESULTS_DIR / f"{tag}_events.csv", index=False)
    pd.Series(out["summary"]).to_csv(config.RESULTS_DIR / f"{tag}_summary.csv", header=["value"])
    print(f"\nSaved tables to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
