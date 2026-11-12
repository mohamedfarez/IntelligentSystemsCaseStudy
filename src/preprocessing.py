"""Preprocessing pipeline shared by the learned detectors.

Steps (in this order):

1. **Mask frozen segments** — records inside flat-line logging segments
   (all analogue signals unchanged for >= 30 min, see
   :func:`src.data_loading.find_flatlines`) are removed, because they show a
   stuck logger rather than machine behaviour.
2. **Gap-aware windows** — the remaining records are aggregated into fixed
   windows indexed by their END time. Windows with too few records
   (recording gaps, removed frozen records) are dropped instead of being
   filled, so no artificial values are created.
3. **Chronological split** — windows ending at or before ``TRAIN_END``
   (February 2020) are normal data. Their last ``val_days`` days form a
   validation set (used to set alarm thresholds and, for neural models, to
   stop training); the earlier days form the training set. Everything after
   ``TRAIN_END`` is the test set.
4. **Scaling fitted on training data only** — the scaler never sees
   validation or test windows, which avoids information leakage.

The LPS signal is *not* used as an input feature: it is the low-pressure
symptom of a developing failure and serves as the reference point for early
warning in the evaluation (:func:`src.evaluation.lps_onsets`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loading import find_flatlines
from src.evaluation import warning_window_labels

#: Digital signals used as features (share of active time per window).
#: LPS is excluded on purpose (evaluation reference, see module docstring).
DIGITAL_FEATURE_COLS: list[str] = [c for c in config.DIGITAL_COLS if c != "LPS"]

#: Statistics computed for every analogue signal in each window.
ANALOG_STATS: list[str] = ["mean", "std", "min", "max"]


@dataclass
class PreprocessConfig:
    """Parameters of the preprocessing pipeline.

    Attributes:
        window: Aggregation window (pandas offset alias).
        min_coverage: Minimum share of the expected number of records a
            window must contain to be kept (expected = window / 10 s).
        mask_frozen: Remove records inside frozen logging segments.
        load_current_a: Motor current above which the compressor is
            considered to run under load.
        train_end: Boundary between normal (training + validation) and test period.
        val_days: Length of the validation set at the end of the normal period.
        exclude_horizon: Training windows this close to a reported failure
            are removed (none fall in February, kept as a safeguard).
    """

    window: str = "10min"
    min_coverage: float = 0.5
    mask_frozen: bool = True
    load_current_a: float = config.MOTOR_LOAD_CURRENT_A
    train_end: pd.Timestamp = config.TRAIN_END
    val_days: int = 7
    exclude_horizon: str = "2h"


# ---------------------------------------------------------------------------
# Step 1: frozen segments
# ---------------------------------------------------------------------------
def frozen_mask(df: pd.DataFrame, segments: pd.DataFrame | None = None) -> pd.Series:
    """Mark records that lie inside a frozen logging segment.

    Args:
        df: Raw time-indexed frame.
        segments: Output of :func:`find_flatlines`; computed if None.

    Returns:
        Boolean series aligned to ``df.index`` (True = frozen record).
    """
    segments = find_flatlines(df) if segments is None else segments
    mask = np.zeros(len(df), dtype=bool)
    for start, end in zip(segments["start"], segments["end"]):
        mask |= (df.index >= start) & (df.index <= end)
    return pd.Series(mask, index=df.index, name="is_frozen")


def remove_frozen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop records inside frozen segments.

    Returns:
        ``(clean_df, segments)`` — the filtered frame and the removed segments.
    """
    segments = find_flatlines(df)
    mask = frozen_mask(df, segments)
    return df.loc[~mask.to_numpy()], segments


# ---------------------------------------------------------------------------
# Step 2: window features
# ---------------------------------------------------------------------------
def window_features(df: pd.DataFrame, window: str = "10min",
                    min_coverage: float = 0.5,
                    load_current_a: float = config.MOTOR_LOAD_CURRENT_A) -> pd.DataFrame:
    """Aggregate records into gap-aware windows indexed by their END time.

    Features per window: mean, standard deviation, minimum and maximum of
    every analogue signal; share of active time of every digital signal
    except LPS; share of time under load (motor current above
    ``load_current_a``). Windows are left-closed ``[t - window, t)``.

    Args:
        df: Time-indexed frame (ideally after :func:`remove_frozen`).
        window: Pandas offset alias.
        min_coverage: Minimum share of expected records per window.
        load_current_a: Load threshold for the motor current.

    Returns:
        Feature frame indexed by ``window_end``; a ``coverage`` column gives
        the share of expected records found in each window.
    """
    analog = [c for c in config.ANALOG_COLS if c in df.columns]
    digital = [c for c in DIGITAL_FEATURE_COLS if c in df.columns]
    res = df.resample(window)

    parts = []
    stats = res[analog].agg(ANALOG_STATS)
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    parts.append(stats)
    dig = res[digital].mean()
    dig.columns = [f"{c}_active" for c in digital]
    parts.append(dig)
    load = (df["Motor_current"] > load_current_a).astype("float32").resample(window).mean()
    parts.append(load.rename("load_ratio").to_frame())

    expected = pd.Timedelta(window) / config.SAMPLING_INTERVAL
    count = res["Motor_current"].count().rename("coverage") / expected
    parts.append(count.to_frame())

    feats = pd.concat(parts, axis=1)
    feats = feats[feats["coverage"] >= min_coverage].dropna()
    feats.index = feats.index + pd.Timedelta(window)
    feats.index.name = "window_end"
    return feats.astype("float32")


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Model input columns (everything except bookkeeping columns)."""
    return [c for c in feats.columns if c != "coverage"]


# ---------------------------------------------------------------------------
# Steps 3-4: split and scaling
# ---------------------------------------------------------------------------
def chronological_split(feats: pd.DataFrame, train_end: pd.Timestamp = config.TRAIN_END,
                        exclude_horizon: str = "2h",
                        val_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split windows in time into training, validation and test sets.

    Args:
        feats: Window features indexed by window end.
        train_end: Last window end of the normal period.
        exclude_horizon: Warning horizon used to drop normal-period windows
            near reported failures (none fall in February; a safeguard).
        val_days: Days at the end of the normal period used for validation.

    Returns:
        ``(train, val, test)``.

    Raises:
        ValueError: If any part is empty.
    """
    normal = feats[feats.index <= train_end]
    normal = normal[~warning_window_labels(normal.index, exclude_horizon).to_numpy()]
    val_start = normal.index.max() - pd.Timedelta(days=val_days)
    train = normal[normal.index <= val_start]
    val = normal[normal.index > val_start]
    test = feats[feats.index > train_end]
    if train.empty or val.empty or test.empty:
        raise ValueError("Empty training, validation or test set - check TRAIN_END, val_days and the data range.")
    return train, val, test


class TrainOnlyScaler:
    """Standardisation whose mean and variance come from training windows only."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.columns_: list[str] = []

    def fit(self, train: pd.DataFrame) -> "TrainOnlyScaler":
        """Estimate mean and standard deviation per feature on ``train``."""
        self.columns_ = feature_columns(train)
        self.scaler.fit(train[self.columns_].to_numpy())
        return self

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Scale ``feats`` with the training statistics.

        Features that are constant in training (standard deviation 0) are
        only centred, as done by scikit-learn.
        """
        values = self.scaler.transform(feats[self.columns_].to_numpy())
        return pd.DataFrame(values, index=feats.index, columns=self.columns_).astype("float32")


@dataclass
class PreparedData:
    """Output of :func:`prepare`."""

    features: pd.DataFrame          # all windows, unscaled
    train: pd.DataFrame             # scaled training windows
    val: pd.DataFrame               # scaled validation windows (end of February)
    test: pd.DataFrame              # scaled test windows
    scaler: TrainOnlyScaler
    frozen_segments: pd.DataFrame   # segments removed in step 1
    n_records_removed: int


def prepare(df: pd.DataFrame, cfg: PreprocessConfig | None = None) -> PreparedData:
    """Run the full preprocessing pipeline (steps 1-4).

    Args:
        df: Raw time-indexed frame from :func:`src.data_loading.load_metropt3`.
        cfg: Pipeline parameters.

    Returns:
        A :class:`PreparedData` bundle.
    """
    cfg = cfg or PreprocessConfig()
    if cfg.mask_frozen:
        clean, segments = remove_frozen(df)
    else:
        clean, segments = df, find_flatlines(df).iloc[0:0]
    feats = window_features(clean, cfg.window, cfg.min_coverage, cfg.load_current_a)
    train, val, test = chronological_split(feats, cfg.train_end, cfg.exclude_horizon, cfg.val_days)
    scaler = TrainOnlyScaler().fit(train)
    return PreparedData(
        features=feats,
        train=scaler.transform(train),
        val=scaler.transform(val),
        test=scaler.transform(test),
        scaler=scaler,
        frozen_segments=segments,
        n_records_removed=len(df) - len(clean),
    )
