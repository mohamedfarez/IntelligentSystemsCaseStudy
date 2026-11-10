# Early Warning of Air-Leak Failures in Metro Compressors: Explainable Anomaly Detection on the MetroPT-3 Dataset

Case study for the module **MDM2 – Intelligent Systems in Production**, M.Eng. Applied AI for Digital Production Management, Technische Hochschule Deggendorf (winter semester 2026/27).

**Team (Group 6):** Fuad Obiedat · Abdennour Djeha · Mohamed Fares · Hady Nasser
**Contact:** mohamed.fares@stud.th-deg.de

---

## 1. Problem

The Air Production Unit (APU) of a Metro do Porto train supplies compressed air to pneumatic subsystems such as the secondary suspension. The trains have no backup APU, so an unplanned failure — here, recurring air leaks — takes the vehicle out of service. Maintenance is currently performed at fixed intervals.

**Objective:** build an explainable early-warning model that flags abnormal APU behaviour *before* a failure, supporting a move from fixed-interval to condition-based maintenance. Success is measured with operational, event-level KPIs: failures detected (event recall), warning lead time and false alarms.

## 2. Dataset

**MetroPT-3** — 1,516,948 records of 15 signals (7 analogue, 8 digital) from the APU of one train, February–August 2020, sampled every 10 s. The data is unlabelled; ground truth consists of four maintenance-reported air-leak failures. See [`data/README.md`](data/README.md) for the signal list, failure periods and data-quality results.

> Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R

Licence: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The raw data is **not** redistributed in this repository; it is downloaded with `scripts/download_data.py`.

## 3. Method overview

1. **Data quality and EDA** — missing values, duplicate timestamps, sampling interval, recording gaps, signal behaviour around the four failures, compressor duty cycle.
2. **Chronological split** — the first month (February 2020, normal operation only) is used for training/calibration; the remaining months are the test period. No random shuffling, so no information from the future leaks into training.
3. **Threshold baseline** — signals are aggregated into 10-minute windows (minimum `TP3` and `Reservoirs` pressure, share of time the motor runs under load (> 4.7 A), mean motor current). Each window is time-stamped at its end, when its data is complete. Thresholds are training-period quantiles; an alarm requires at least two rules firing in three consecutive windows without a recording gap in between.
4. **Event-level evaluation** — a failure is detected if an alarm occurs between *failure start − warning horizon* (provisionally 2 h) and *failure end*; lead time in minutes (negative = detection delay), false-alarm episodes per day and window-level precision/recall/F1 are reported. With only four failures, results are reported with their uncertainty in mind rather than as general performance claims.

### Key EDA findings

- No missing values or duplicate timestamps, but 275 recording gaps (> 5 min) covering 17.7 % of the period.
- 9 frozen logging segments (all analogue signals constant for ≥ 30 min), 7 days 2 hours in total (3.3 % of records), including the hours directly before failure 1 (17 April 09:20 – 18 April 00:18).
- `Motor_current` has three states (~0 / ~3.8 / ~5.4–6.3 A). Hourly share of time under load: 0.975 during failures, 0.161 in the 24 h before a failure, 0.145 in normal operation — leaks are easy to see once they happen, but hard to anticipate.
- February (training month) differs from March–August (lower duty cycle and oil temperature), a distribution shift that affects any model trained on February alone.

### Current results: threshold baseline (test period March–August 2020)

| Failure | Reported start | First alarm | Lead time |
|---|---|---|---|
| 1 | 2020-04-18 00:00 | 2020-04-18 00:50 | −50 min (late) |
| 2 | 2020-05-29 23:30 | 2020-05-29 23:50 | −20 min (late) |
| 3 | 2020-06-05 10:00 | 2020-06-05 10:20 | −20 min (late) |
| 4 | 2020-07-15 14:30 | 2020-07-15 12:30 | +120 min (early) |

| KPI | Value |
|---|---|
| Failures detected | 4 / 4 |
| Early warnings (before reported start) | 1 / 4 |
| False-alarm episodes | 45 (≈ 0.24 per day) |
| Window-level precision / recall / F1 | 0.32 / 0.91 / 0.48 |

Settings: 10-min windows, thresholds = 1 %/99 % quantiles of February, ≥ 2 rules in 3 consecutive windows, 2-h warning horizon. The baseline detects ongoing leaks but gives almost no early warning; the longest false-alarm block (22–25 June) falls inside a frozen logging segment. Source: `notebooks/01_EDA.ipynb`, `results/baseline_threshold_*.csv`.

### Isolation Forest prototype (methodology phase 1)

Shared pipeline: frozen segments removed (9 segments, 3.3 % of records), gap-aware 10-min windows with 36 features (LPS excluded), training on February only, train-only scaling. Isolation Forest (200 trees, sub-sample 256, seed 42) with a common alarm logic: threshold Q3 + 3·IQR of the training scores and a low-pass filter (α = 0.3, alarm above 0.5).

| KPI | Threshold rule | Isolation Forest | IF without frozen mask |
|---|---|---|---|
| Failures detected | 4 / 4 | 4 / 4 | 4 / 4 |
| Early warnings | 1 / 4 | 1 / 4 | 1 / 4 |
| Median lead time | −20 min | −5 min | −5 min |
| False alarms per day | 0.245 | 0.522 | 0.543 |
| Window precision / recall / F1 | 0.32 / 0.91 / 0.48 | 0.42 / 0.95 / 0.58 | 0.29 / 0.92 / 0.44 |

The Isolation Forest reacts 10–20 min earlier than the threshold rule for failures 1–3 but raises about twice as many false alarms, mainly because the test months differ from February. The low-pressure switch (LPS) did not activate near failures 1 and 2. Source: `notebooks/02_preprocessing_iforest.ipynb`, `results/detector_comparison.csv`.

The LSTM autoencoder and SHAP explanations follow in methodology phase 2 (see Section 6).

## 4. Repository structure

```
IntelligentSystemsCaseStudy/
├── README.md                     # this file
├── requirements.txt              # pinned dependencies
├── .gitignore
├── data/
│   └── README.md                 # dataset source, licence, download, ground truth
├── scripts/
│   └── download_data.py          # downloads MetroPT-3 from UCI into data/raw/
├── src/
│   ├── __init__.py
│   ├── config.py                 # paths, random seed, schema, failure periods, split date
│   ├── data_loading.py           # CSV loading, timestamp parsing, quality checks
│   ├── preprocessing.py          # frozen-segment mask, gap-aware windows, split, train-only scaling
│   ├── threshold_baseline.py     # rule-based baseline
│   ├── detectors.py              # learned detectors (Isolation Forest)
│   ├── alarm_logic.py            # Q3 + 3·IQR threshold and low-pass filter
│   ├── evaluation.py             # shared event-level evaluation
│   └── run_experiment.py         # end-to-end run of a learned detector
├── notebooks/
│   ├── 01_EDA.ipynb              # exploratory data analysis
│   └── 02_preprocessing_iforest.ipynb  # preprocessing and Isolation Forest prototype
├── results/
│   └── figures/                  # figures generated by the notebooks
│       ├── EDA/                  # exploratory data analysis (notebook 01)
│       ├── baseline/             # threshold baseline (notebook 01)
│       ├── preprocessing/        # preprocessing checks (notebook 02)
│       └── isolation_forest/     # Isolation Forest results (notebook 02)
└── docs/
    ├── 01_Proposal/              # one-page proposal (PDF) and proposal slides
    ├── 02_Bibliography_Report/   # literature review (milestone 2)
    └── 03_Methodology_Phase1/    # theoretical framework and system design (milestone 3)
```

### Documents

| Milestone | Document |
|---|---|
| Proposal (2026-10-20) | [One-page proposal (PDF)](docs/01_Proposal/MDM2104_Proposal_MetroPT3_Group6.pdf) · [Slides (PPTX)](docs/01_Proposal/Group6_Proposal.pptx) |
| Bibliography report (2026-11-03) | [Bibliography report (PDF)](docs/02_Bibliography_Report/Group6_Bibliography_Report.pdf) · [LaTeX sources](docs/02_Bibliography_Report/source/) |
| Methodology, phase 1 (2026-11-24) | [Methodology report (PDF)](docs/03_Methodology_Phase1/Group6_Methodology_Phase1.pdf) · [LaTeX sources](docs/03_Methodology_Phase1/source/) |

## 5. Setup and reproduction

Requires Python ≥ 3.10. Both notebooks also run on Google Colab: their first cell clones this repository and downloads the data.

```bash
git clone https://github.com/mohamedfarez/IntelligentSystemsCaseStudy.git
cd IntelligentSystemsCaseStudy

python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 1) Download the dataset (~208 MB) into data/raw/
python scripts/download_data.py

# 2) Data quality report
python -m src.data_loading

# 3) Exploratory data analysis (figures are written to results/figures/EDA/ and baseline/)
jupyter notebook notebooks/01_EDA.ipynb

# 4) Threshold baseline (tables are written to results/)
python -m src.threshold_baseline
python -m src.threshold_baseline --window 5min --min-rules 1 --persistence 2 --horizon 6h

# 5) Preprocessing and Isolation Forest (tables are written to results/)
python -m src.run_experiment --detector iforest
python -m src.run_experiment --detector iforest --no-frozen-mask
jupyter notebook notebooks/02_preprocessing_iforest.ipynb
```

All random operations use the fixed seed in `src/config.py` (`RANDOM_SEED = 42`).

## 6. Milestones

| Date | Milestone | Deliverables | Status |
|---|---|---|---|
| 2026-10-20 | Proposal | One-page proposal, proposal slides (`docs/01_Proposal/`) | ✅ Done |
| 2026-11-03 | Literature map & method outline | Bibliography report, data download, EDA, threshold baseline | ✅ Done |
| 2026-11-24 | Methodology, phase 1 | Theoretical framework, system design, preprocessing, Isolation Forest prototype | ✅ Done |
| 2026-12-08 | Methodology, phase 2 | LSTM autoencoder, SHAP module, experimental design | ⏳ Planned |
| 2026-12-22 | Demo; experiments & analysis | Event-level evaluation, SHAP explanations | ⏳ Planned |
| 2027-01-12 | Final presentation & report | Final report, presentation, handout, demo video | ⏳ Planned |

## 7. Key references

1. Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021). Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry. In *2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA)*. IEEE. https://doi.org/10.1109/DSAA53316.2021.9564181
2. Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data, 9*, 764. https://doi.org/10.1038/s41597-022-01877-3
3. Jakobs, M., Veloso, B., & Gama, J. (2025). Interpretable rules for online failure prediction: A case study on the Metro do Porto dataset. *arXiv preprint* arXiv:2502.07394. https://arxiv.org/abs/2502.07394
4. Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R

## Licence

Code in this repository is provided for academic purposes within the MDM2 case study. The MetroPT-3 data remains under its original CC BY 4.0 licence and must be cited as above.
