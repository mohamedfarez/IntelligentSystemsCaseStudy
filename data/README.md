# Data

This folder holds the MetroPT-3 dataset **locally only**. `data/raw/` and `data/processed/` are listed in `.gitignore`; the raw data is never committed to this repository.

## Source

| Item | Value |
|---|---|
| Dataset | MetroPT-3 Dataset (UCI Machine Learning Repository, ID 791) |
| DOI | https://doi.org/10.24432/C5VW3R |
| Creators | N. Davari, B. Veloso, R. Ribeiro, J. Gama |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Archive | https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip (~208 MB) |
| File used | `MetroPT3(AirCompressor).csv` |
| Instances | 1,516,948 (as stated by UCI) |
| Period | 2020-02-01 00:00:00 – 2020-09-01 03:59:50 |
| Sampling | median interval 10 s (0.1 Hz); the UCI page states both 1 Hz and 0.1 Hz, the data confirms 0.1 Hz |

**Citation**

> Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R

The dataset is described in detail in: Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data, 9*, 764. https://doi.org/10.1038/s41597-022-01877-3

## Download

From the repository root:

```bash
python scripts/download_data.py           # downloads and extracts into data/raw/
python scripts/download_data.py --force   # re-download
```

The script streams the archive, extracts only the CSV, deletes the ZIP (use `--keep-zip` to keep it) and checks the row count against the UCI description.

## Signals

Canonical column names used in the code (`src/config.py`). Raw spellings such as `DV_eletric` are mapped automatically by `src/data_loading.py`.

| Type | Columns |
|---|---|
| Analogue (7) | `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs` (bar); `Oil_temperature` (°C); `Motor_current` (A) |
| Digital (8) | `COMP`, `DV_electric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses` |

`Motor_current` takes three states in the data: ~0 A (off), ~3.8 A (offloaded) and ~5.4–6.3 A (under load). These differ from the 4 A / 7 A levels in the UCI description; the code uses 4.7 A to separate offloaded from loaded operation (`config.MOTOR_LOAD_CURRENT_A`).

## Data quality (from `notebooks/01_EDA.ipynb`)

| Check | Result |
|---|---|
| Records | 1,516,948 (matches UCI) |
| Missing values | 0 |
| Duplicate timestamps | 0 |
| Recording gaps > 5 min | 275, together 37 days 18 h (17.7 % of the period); longest 2020-04-25 → 2020-04-27 (2 days) |
| 29 February 2020 | not recorded (gap 2020-02-28 23:57 → 2020-03-01 04:00) |
| Frozen logging segments | 9 segments with all analogue signals constant for ≥ 30 min, 7 days 2 h in total (50,716 records, 3.3 %), between 2020-04-13 and 2020-07-22; includes 2020-04-17 09:20 → 2020-04-18 00:18 (directly before failure 1) and 2020-06-22 → 2020-06-25 (longest, 2.6 days). Listed by `src.data_loading.find_flatlines` |

Run `python -m src.data_loading` to reproduce this report.

## Ground truth

The data is unlabelled. The only ground truth is the list of maintenance-reported failures published on the UCI dataset page (all air leaks), stored in `src/config.py` (`FAILURE_PERIODS`):

| # | Start | End | Type |
|---|---|---|---|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak |

## Folder layout (local)

```
data/
├── README.md        # this file (tracked)
├── raw/             # downloaded CSV (ignored by git)
└── processed/       # derived data, e.g. windowed features (ignored by git)
```
