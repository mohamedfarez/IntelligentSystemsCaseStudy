"""Common alarm logic: from a continuous anomaly score to a binary alarm.

All learned detectors (Isolation Forest now, LSTM autoencoder later) output
one anomaly score per window, where higher means more unusual. The same two
steps turn every score into alarms, so differences between detectors come
from the scores and not from different alarm rules:

1. **Threshold from validation scores** — ``Q3 + k * IQR`` of the scores on
   the normal validation windows (last week of February, not used to fit the
   detector), with ``k = 3`` as in Silva et al. (2023). Scores on the
   detector's own training data are optimistic (too low), which would give
   a threshold that is too low. No test data is used.
2. **Low-pass filter** — single windows above the threshold are often
   noise. The binary exceedance signal ``x_i`` is smoothed with the filter
   of Ribeiro et al. (2016), ``y_i = y_{i-1} + alpha * (x_i - y_{i-1})``,
   and an alarm is raised while ``y_i > 0.5``. With ``alpha = 0.3`` two
   consecutive exceeding windows are needed for a first alarm.

The filter state is reset to 0 after a recording gap, so windows on both
sides of a gap are never treated as consecutive and no values are filled in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlarmConfig:
    """Parameters of the alarm logic.

    Attributes:
        iqr_factor: ``k`` in ``Q3 + k * IQR``.
        alpha: Smoothing factor of the low-pass filter (0 < alpha <= 1).
        level: Alarm level of the filtered signal.
        window: Window length, used to detect recording gaps.
    """

    iqr_factor: float = 3.0
    alpha: float = 0.3
    level: float = 0.5
    window: str = "10min"


def iqr_threshold(train_scores: pd.Series | np.ndarray, k: float = 3.0) -> float:
    """Upper outlier fence ``Q3 + k * IQR`` of the training scores."""
    q1, q3 = np.quantile(np.asarray(train_scores, dtype=float), [0.25, 0.75])
    return float(q3 + k * (q3 - q1))


def low_pass(exceed: pd.Series, alpha: float = 0.3, window: str = "10min") -> pd.Series:
    """Exponential low-pass filter of a boolean exceedance series.

    Args:
        exceed: Boolean series indexed by window end time.
        alpha: Smoothing factor.
        window: Expected spacing of windows; a larger spacing is a gap and
            resets the filter.

    Returns:
        Filtered series with values between 0 and 1.
    """
    x = exceed.to_numpy(dtype=float)
    gap = pd.Series(exceed.index).diff().to_numpy() != pd.Timedelta(window)
    y = np.empty_like(x)
    prev = 0.0
    for i in range(len(x)):
        if gap[i]:
            prev = 0.0
        prev = prev + alpha * (x[i] - prev)
        y[i] = prev
    return pd.Series(y, index=exceed.index, name="filtered")


class AlarmRule:
    """Threshold + low-pass filter, fitted on normal validation scores only."""

    def __init__(self, cfg: AlarmConfig | None = None) -> None:
        self.cfg = cfg or AlarmConfig()
        self.threshold_: float | None = None

    def fit(self, normal_scores: pd.Series) -> "AlarmRule":
        """Set the threshold from scores on normal (validation) windows."""
        self.threshold_ = iqr_threshold(normal_scores, self.cfg.iqr_factor)
        return self

    def predict(self, scores: pd.Series) -> pd.Series:
        """Return the boolean alarm series for ``scores``."""
        if self.threshold_ is None:
            raise RuntimeError("Call fit() before predict().")
        filtered = low_pass(scores > self.threshold_, self.cfg.alpha, self.cfg.window)
        return (filtered > self.cfg.level).rename("alarm")
