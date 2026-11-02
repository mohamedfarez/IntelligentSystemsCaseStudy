"""Loading, cleaning and basic quality checks for the MetroPT-3 CSV.

Typical use::

    from src.data_loading import load_metropt3, quality_report
    df = load_metropt3()
    report = quality_report(df)
    print(report.summary())

Run as a module to print the quality report of the downloaded file::

    python -m src.data_loading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV columns to the canonical names defined in ``config``.

    The raw file contains an unnamed index column and some spellings that
    differ from the documentation (e.g. ``DV_eletric``). Matching is
    case-insensitive; unknown columns are kept unchanged.

    Args:
        df: Raw data frame as read from the CSV.

    Returns:
        A new data frame with canonical column names and without the
        unnamed index column.
    """
    canonical = {c.lower(): c for c in [config.TIMESTAMP_COL, *config.SENSOR_COLS]}
    canonical.update(config.COLUMN_ALIASES)

    out = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    rename = {c: canonical[str(c).strip().lower()]
              for c in out.columns if str(c).strip().lower() in canonical}
    return out.rename(columns=rename)


def load_metropt3(
    path: Path | str | None = None,
    nrows: int | None = None,
    drop_duplicate_timestamps: bool = True,
) -> pd.DataFrame:
    """Load MetroPT-3 into a time-indexed data frame.

    Steps: read CSV, standardise column names, parse timestamps, sort
    chronologically, optionally drop exact duplicate timestamps (first
    occurrence kept) and cast sensors to ``float32`` to save memory.

    Args:
        path: CSV path. Defaults to ``config.RAW_CSV_PATH``.
        nrows: Read only the first ``nrows`` rows (useful for quick tests).
        drop_duplicate_timestamps: If True, keep only the first record of
            each duplicated timestamp.

    Returns:
        Data frame indexed by a sorted ``DatetimeIndex`` named ``timestamp``
        with one column per sensor in ``config.SENSOR_COLS`` (if present).

    Raises:
        FileNotFoundError: If the CSV does not exist (run
            ``python scripts/download_data.py`` first).
        KeyError: If the timestamp column is missing.
    """
    csv_path = Path(path) if path is not None else config.RAW_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python scripts/download_data.py` first."
        )

    df = standardize_columns(pd.read_csv(csv_path, nrows=nrows))
    if config.TIMESTAMP_COL not in df.columns:
        raise KeyError(f"Column '{config.TIMESTAMP_COL}' not found in {csv_path}.")

    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
    df = df.sort_values(config.TIMESTAMP_COL, kind="stable")
    if drop_duplicate_timestamps:
        df = df.drop_duplicates(subset=config.TIMESTAMP_COL, keep="first")
    df = df.set_index(config.TIMESTAMP_COL)

    missing = [c for c in config.SENSOR_COLS if c not in df.columns]
    if missing:
        logger.warning("Expected sensor columns not found: %s", missing)
    sensors = [c for c in config.SENSOR_COLS if c in df.columns]
    return df[sensors].astype("float32")


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------
def failure_mask(index: pd.DatetimeIndex,
                 periods: pd.DataFrame = config.FAILURE_PERIODS) -> pd.Series:
    """Mark timestamps that fall inside a reported failure period.

    Args:
        index: Timestamps to label.
        periods: Data frame with ``start`` and ``end`` columns (inclusive).

    Returns:
        Boolean series aligned to ``index`` (True = inside a failure period).
    """
    mask = np.zeros(len(index), dtype=bool)
    for start, end in zip(periods["start"], periods["end"]):
        mask |= (index >= start) & (index <= end)
    return pd.Series(mask, index=index, name="is_failure")


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def find_gaps(index: pd.DatetimeIndex, min_gap: str | pd.Timedelta = "5min") -> pd.DataFrame:
    """List recording gaps longer than ``min_gap``.

    Args:
        index: Sorted timestamps.
        min_gap: Minimum gap length to report.

    Returns:
        Data frame with ``gap_start`` (last sample before the gap),
        ``gap_end`` (first sample after it) and ``duration``, sorted by
        duration in descending order.
    """
    threshold = pd.Timedelta(min_gap)
    ts = pd.Series(index)
    delta = ts.diff()
    is_gap = delta > threshold
    gaps = pd.DataFrame({
        "gap_start": ts.shift(1)[is_gap].to_numpy(),
        "gap_end": ts[is_gap].to_numpy(),
        "duration": delta[is_gap].to_numpy(),
    })
    return gaps.sort_values("duration", ascending=False).reset_index(drop=True)


def find_flatlines(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    min_duration: str | pd.Timedelta = config.FLATLINE_MIN_DURATION,
    max_step: str | pd.Timedelta = "5min",
) -> pd.DataFrame:
    """Find frozen logging segments where all selected signals stay constant.

    A segment is a run of consecutive records in which *every* column in
    ``cols`` keeps exactly the same value as in the previous record. During
    normal operation at least one analogue signal (pressures, oil
    temperature) changes within minutes, so long runs indicate a stuck data
    logger rather than real machine behaviour. Runs are split at recording
    gaps longer than ``max_step``.

    Args:
        df: Time-indexed frame from :func:`load_metropt3`.
        cols: Columns to check; defaults to the analogue signals.
        min_duration: Minimum segment duration to report.
        max_step: Time step above which a run is split (recording gap).

    Returns:
        Data frame with ``start``, ``end``, ``duration`` and ``n_records`` per
        segment, sorted by start time.
    """
    cols = cols or [c for c in config.ANALOG_COLS if c in df.columns]
    values = df[cols]
    unchanged = (values.diff().abs().sum(axis=1) == 0) & values.notna().all(axis=1)
    contiguous = pd.Series(df.index, index=df.index).diff() <= pd.Timedelta(max_step)
    same_run = unchanged & contiguous
    run_id = (~same_run).cumsum()

    ts = pd.Series(df.index, index=df.index)
    runs = ts.groupby(run_id.to_numpy()).agg(["min", "max", "count"])
    runs.columns = ["start", "end", "n_records"]
    runs["duration"] = runs["end"] - runs["start"]
    runs = runs[runs["duration"] >= pd.Timedelta(min_duration)]
    return runs[["start", "end", "duration", "n_records"]].sort_values("start").reset_index(drop=True)


@dataclass
class QualityReport:
    """Container for the basic data quality checks."""

    n_rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_duplicate_timestamps: int
    missing_per_column: pd.Series
    sampling_interval_median: pd.Timedelta
    sampling_interval_counts: pd.Series
    gaps: pd.DataFrame
    flatlines: pd.DataFrame
    constant_columns: list[str] = field(default_factory=list)
    digital_non_binary: dict[str, list[float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a short, human-readable text summary."""
        total_gap = self.gaps["duration"].sum() if len(self.gaps) else pd.Timedelta(0)
        total_flat = self.flatlines["duration"].sum() if len(self.flatlines) else pd.Timedelta(0)
        span = self.end - self.start
        lines = [
            f"Rows:                     {self.n_rows:,}",
            f"Period:                   {self.start} -> {self.end}",
            f"Duplicate timestamps:     {self.n_duplicate_timestamps:,}",
            f"Missing values (total):   {int(self.missing_per_column.sum()):,}",
            f"Median sampling interval: {self.sampling_interval_median}",
            f"Gaps > threshold:         {len(self.gaps):,} (total {total_gap}, "
            f"{100 * (total_gap / span):.1f} % of the period)",
            f"Frozen (flat-line) segs:  {len(self.flatlines):,} (total {total_flat}, "
            f"{self.flatlines['n_records'].sum() if len(self.flatlines) else 0:,} records)",
            f"Constant columns:         {self.constant_columns or 'none'}",
            f"Non-binary digital cols:  {list(self.digital_non_binary) or 'none'}",
        ]
        return "\n".join(lines)


def quality_report(df: pd.DataFrame, min_gap: str = "5min",
                   raw_index: pd.DatetimeIndex | None = None) -> QualityReport:
    """Run basic quality checks on a loaded MetroPT-3 frame.

    Args:
        df: Output of :func:`load_metropt3`.
        min_gap: Minimum duration reported as a recording gap.
        raw_index: Optional timestamps *before* de-duplication, used to count
            duplicates. If None, duplicates are counted on ``df.index``.

    Returns:
        A :class:`QualityReport`.
    """
    idx = df.index
    dup_source = raw_index if raw_index is not None else idx
    intervals = pd.Series(idx).diff().dropna()

    digital = [c for c in config.DIGITAL_COLS if c in df.columns]
    non_binary: dict[str, list[float]] = {}
    for col in digital:
        values = df[col].dropna().unique()
        extra = sorted(float(v) for v in values if v not in (0.0, 1.0))
        if extra:
            non_binary[col] = extra[:10]

    return QualityReport(
        n_rows=len(df),
        start=idx.min(),
        end=idx.max(),
        n_duplicate_timestamps=int(pd.Index(dup_source).duplicated().sum()),
        missing_per_column=df.isna().sum(),
        sampling_interval_median=intervals.median(),
        sampling_interval_counts=intervals.value_counts().head(10),
        gaps=find_gaps(idx, min_gap),
        flatlines=find_flatlines(df),
        constant_columns=[c for c in df.columns if df[c].nunique(dropna=True) <= 1],
        digital_non_binary=non_binary,
    )


def main() -> None:
    """Print the quality report for the downloaded dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw = pd.read_csv(config.RAW_CSV_PATH, usecols=lambda c: c.lower() == "timestamp")
    raw_index = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]))
    df = load_metropt3()
    report = quality_report(df, raw_index=raw_index)
    print(report.summary())
    if report.n_rows + report.n_duplicate_timestamps != config.EXPECTED_N_ROWS:
        logger.warning("Row count differs from the %s instances stated by UCI.",
                       f"{config.EXPECTED_N_ROWS:,}")
    print("\nLongest gaps:\n", report.gaps.head(10).to_string(index=False))
    print("\nFrozen (flat-line) segments:\n", report.flatlines.to_string(index=False))


if __name__ == "__main__":
    main()
