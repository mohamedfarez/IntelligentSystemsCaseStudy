"""Central configuration for the MetroPT-3 case study.

All paths, the global random seed, the dataset schema and the officially
reported failure periods are defined here so that every script and notebook
uses the same values.

Sources
-------
* Dataset page, variable descriptions and failure periods:
  UCI Machine Learning Repository, MetroPT-3 Dataset (ID 791),
  https://doi.org/10.24432/C5VW3R (CC BY 4.0).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"

#: Sub-folder of FIGURES_DIR for each figure-name prefix.
FIGURE_SUBDIRS: Final[dict[str, str]] = {
    "eda_": "EDA",
    "baseline_": "baseline",
    "prep_": "preprocessing",
    "iforest_": "isolation_forest",
}


def figure_path(name: str) -> Path:
    """Path of a figure file, sorted into a sub-folder by its name prefix.

    Example: ``figure_path("eda_01_records_per_day")`` ->
    ``results/figures/EDA/eda_01_records_per_day.png``. The folder is created
    if needed; names without a known prefix go directly into FIGURES_DIR.
    """
    folder = FIGURES_DIR
    for prefix, sub in FIGURE_SUBDIRS.items():
        if name.startswith(prefix):
            folder = FIGURES_DIR / sub
            break
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.png"

# ---------------------------------------------------------------------------
# Dataset source
# ---------------------------------------------------------------------------
DATASET_URL: Final[str] = (
    "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
)
DATASET_DOI: Final[str] = "https://doi.org/10.24432/C5VW3R"
DATASET_ZIP_NAME: Final[str] = "metropt+3+dataset.zip"
RAW_CSV_NAME: Final[str] = "MetroPT3(AirCompressor).csv"
RAW_CSV_PATH: Final[Path] = RAW_DATA_DIR / RAW_CSV_NAME
EXPECTED_N_ROWS: Final[int] = 1_516_948  # number of instances stated by UCI

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and the hash seed so that runs are repeatable.

    Args:
        seed: Integer seed applied to all supported random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
TIMESTAMP_COL: Final[str] = "timestamp"

#: Analogue sensors (pressures in bar, current in A, temperature in degC).
ANALOG_COLS: Final[list[str]] = [
    "TP2",              # pressure on the compressor [bar]
    "TP3",              # pressure generated at the pneumatic panel [bar]
    "H1",               # pressure drop at cyclonic separator filter discharge [bar]
    "DV_pressure",      # pressure drop at air-dryer tower discharge [bar]
    "Reservoirs",       # downstream reservoir pressure [bar]
    "Oil_temperature",  # compressor oil temperature [degC]
    "Motor_current",    # current of one motor phase [A]
]

#: Digital (binary) control and status signals.
DIGITAL_COLS: Final[list[str]] = [
    "COMP",             # air intake valve; active when there is no air intake
    "DV_electric",      # compressor outlet valve; active under load
    "Towers",           # active air-drying tower (0 = tower one, 1 = tower two)
    "MPG",              # starts compressor under load when pressure < 8.2 bar
    "LPS",              # low-pressure switch; active when pressure < 7 bar
    "Pressure_switch",  # discharge detected in the air-drying towers
    "Oil_level",        # active when oil is below the expected level
    "Caudal_impulses",  # air-flow pulse counter (APU -> reservoirs)
]

SENSOR_COLS: Final[list[str]] = ANALOG_COLS + DIGITAL_COLS

#: Alternative spellings found in the raw CSV / documentation, mapped to the
#: canonical names above. Matching is case-insensitive.
COLUMN_ALIASES: Final[dict[str, str]] = {
    "dv_eletric": "DV_electric",
    "dv_electric": "DV_electric",
    "caudal_impulse": "Caudal_impulses",
    "caudal_impulses": "Caudal_impulses",
    "towers": "Towers",
}

#: Motor current threshold separating "offloaded" from "under load".
#: The UCI description gives ~0 A off, ~4 A offloaded, ~7 A under load.
#: The observed distribution (notebooks/01_EDA.ipynb, figure
#: eda_05_motor_current_hist) shows three clusters at ~0 A, ~3.8 A and
#: ~5.4-6.3 A; 4.7 A lies in the empty gap between the upper two clusters.
MOTOR_LOAD_CURRENT_A: Final[float] = 4.7

#: Raw sampling interval observed in the data (median), and the minimum
#: duration for which all analogue signals must stay exactly constant to be
#: flagged as a frozen (flat-line) logging segment.
SAMPLING_INTERVAL: Final[pd.Timedelta] = pd.Timedelta("10s")
FLATLINE_MIN_DURATION: Final[pd.Timedelta] = pd.Timedelta("30min")

# ---------------------------------------------------------------------------
# Ground truth: maintenance-reported failures (UCI dataset page)
# ---------------------------------------------------------------------------
FAILURE_PERIODS: Final[pd.DataFrame] = pd.DataFrame(
    {
        "failure_id": [1, 2, 3, 4],
        "start": pd.to_datetime(
            ["2020-04-18 00:00", "2020-05-29 23:30", "2020-06-05 10:00", "2020-07-15 14:30"]
        ),
        "end": pd.to_datetime(
            ["2020-04-18 23:59", "2020-05-30 06:00", "2020-06-07 14:30", "2020-07-15 19:00"]
        ),
        "failure_type": ["Air leak"] * 4,
    }
)

# ---------------------------------------------------------------------------
# Chronological split (proposal: first month = normal-only training data)
# ---------------------------------------------------------------------------
TRAIN_END: Final[pd.Timestamp] = pd.Timestamp("2020-03-01 00:00:00")
