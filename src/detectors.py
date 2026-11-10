"""Anomaly detectors that learn normal behaviour from the training windows.

Every detector follows the same small interface so that the experiment
runner can treat them alike:

* ``fit(train)`` — learn from scaled, normal-only training windows;
* ``score(X)``  — return one anomaly score per window (higher = more unusual).

Isolation Forest (Liu et al., 2008) isolates points by random axis-parallel
splits; anomalies need fewer splits and therefore have shorter average path
lengths. scikit-learn's ``score_samples`` returns the negated anomaly score
of the original paper, so we negate it again.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config


@dataclass
class IForestConfig:
    """Hyper-parameters of the Isolation Forest.

    Defaults follow Liu et al. (2008): sub-sample size 256 and enough trees
    for the path lengths to converge (the paper uses 100; we use 200).
    """

    n_estimators: int = 200
    max_samples: int = 256
    max_features: float = 1.0
    random_state: int = config.RANDOM_SEED


class IsolationForestDetector:
    """Wrapper around :class:`sklearn.ensemble.IsolationForest`."""

    name = "isolation_forest"

    def __init__(self, cfg: IForestConfig | None = None) -> None:
        self.cfg = cfg or IForestConfig()
        self.model = IsolationForest(
            n_estimators=self.cfg.n_estimators,
            max_samples=self.cfg.max_samples,
            max_features=self.cfg.max_features,
            random_state=self.cfg.random_state,
            n_jobs=-1,
        )

    def fit(self, train: pd.DataFrame) -> "IsolationForestDetector":
        """Fit on scaled training windows."""
        self.model.fit(train.to_numpy())
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Anomaly score per window (higher = more anomalous)."""
        return pd.Series(-self.model.score_samples(X.to_numpy()), index=X.index, name="score")
