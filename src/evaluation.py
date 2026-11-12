"""Event-level evaluation shared by all detectors.

Every detector in this project produces a boolean alarm series indexed by
window END time. This module turns such a series into the same set of
operational KPIs, so that detectors are compared under one protocol:

* a failure is **detected** if an alarm occurs between
  ``start - horizon`` and the end of the failure;
* **lead time** (minutes) is measured from the first such alarm to the
  reported failure start (positive = before the failure, negative = delay);
* alarm **episodes** (runs of consecutive alarming windows) outside these
  intervals are counted as **false alarms**, also expressed per day of
  *observed* test time (number of windows x window length), so recording
  gaps do not dilute the rate;
* window-level precision / recall / F1 are reported as a secondary view.
  No point adjustment is applied (see Kim et al., 2022).

Optionally, lead time can also be measured against the first activation of
the low-pressure switch (LPS) near each failure, which is the reference used
by Silva et al. (2023) for the "at least 2 h before LPS" criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src import config


def _run_ids(flags: pd.Series, step: pd.Timedelta) -> pd.Series:
    """Label runs of True values that are consecutive in *time*.

    A new run starts at every False value and wherever the time difference
    to the previous window is not exactly ``step`` (a recording gap).
    """
    t = pd.Series(flags.index, index=flags.index)
    breaks = (~flags) | (t.diff() != step)
    return breaks.cumsum()


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


def lps_onsets(df: pd.DataFrame, horizon: str = "2h",
               periods: pd.DataFrame = config.FAILURE_PERIODS) -> pd.Series:
    """First activation of the low-pressure switch (LPS) near each failure.

    Searches ``[start - horizon, end]`` of every failure for the first raw
    record with ``LPS == 1``.

    Args:
        df: Raw time-indexed frame with an ``LPS`` column.
        horizon: How far before the reported start to search.
        periods: Failure periods.

    Returns:
        Series indexed by ``failure_id`` with the onset time (NaT if LPS
        never fired in the interval).
    """
    h = pd.Timedelta(horizon)
    lps_on = df.index[df["LPS"].to_numpy() > 0.5]
    out = {}
    for _, p in periods.iterrows():
        hits = lps_on[(lps_on >= p["start"] - h) & (lps_on <= p["end"])]
        out[p["failure_id"]] = hits.min() if len(hits) else pd.NaT
    return pd.Series(out, name="lps_onset", dtype="datetime64[ns]")


def evaluate_events(alarms: pd.Series, horizon: str, window: str,
                    periods: pd.DataFrame = config.FAILURE_PERIODS,
                    lps: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Event-level evaluation: detection, lead time and false alarms.

    Args:
        alarms: Boolean alarm series (test period only), indexed by window end.
        horizon: Early-warning horizon.
        window: Window length (for episode merging).
        periods: Failure periods to evaluate against.
        lps: Optional output of :func:`lps_onsets`; adds lead time relative
            to the first LPS activation.

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
        row = {
            "failure_id": p["failure_id"],
            "start": p["start"],
            "end": p["end"],
            "detected": len(hits) > 0,
            "first_alarm": first,
            "lead_time_min": (p["start"] - first) / pd.Timedelta("1min") if len(hits) else np.nan,
            "early_warning": bool(len(hits) and first < p["start"]),
        }
        if lps is not None:
            onset = lps.get(p["failure_id"], pd.NaT)
            row["lps_onset"] = onset
            row["lead_vs_lps_min"] = ((onset - first) / pd.Timedelta("1min")
                                      if len(hits) and pd.notna(onset) else np.nan)
        rows.append(row)
    events = pd.DataFrame(rows)

    episodes = alarm_episodes(alarms, window)
    is_true = np.zeros(len(episodes), dtype=bool)
    for _, p in in_test.iterrows():
        is_true |= ((episodes["end"] >= p["start"] - h) & (episodes["start"] <= p["end"])).to_numpy()
    n_false = int((~is_true).sum())
    observed_days = len(alarms) * pd.Timedelta(window) / pd.Timedelta("1D")
    span_days = (alarms.index.max() - alarms.index.min()) / pd.Timedelta("1D")

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
        "false_alarms_per_day": n_false / observed_days if observed_days > 0 else float("nan"),
        "observed_test_days": observed_days,
        "test_span_days": span_days,
        "window_precision": float(prec),
        "window_recall": float(rec),
        "window_f1": float(f1),
    }
    return events, summary
