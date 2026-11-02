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

More advanced detectors and explainability methods are planned for later milestones (see Section 6).

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
│   └── threshold_baseline.py     # rule-based baseline + event-level evaluation
├── notebooks/
│   └── 01_EDA.ipynb              # exploratory data analysis
├── results/
│   └── figures/                  # EDA figures (generated by the notebook)
└── docs/
    ├── 01_Proposal/              # one-page proposal (PDF) and proposal slides
    └── 02_Bibliography_Report/   # literature review (milestone 2)
```

### Documents

| Milestone | Document |
|---|---|
| Proposal (2026-10-20) | [One-page proposal (PDF)](docs/01_Proposal/MDM2104_Proposal_MetroPT3_Group6.pdf) · [Slides (PPTX)](docs/01_Proposal/Group6_Proposal.pptx) |
| Bibliography report (2026-11-03) | [Bibliography report (PDF)](docs/02_Bibliography_Report/Group6_Bibliography_Report.pdf) · [LaTeX sources](docs/02_Bibliography_Report/source/) |

## 5. Setup and reproduction

Requires Python ≥ 3.10. `notebooks/01_EDA.ipynb` can also run on Google Colab: its first cell uploads the `src/` and `scripts/` folders as a zip and downloads the data.

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

# 3) Exploratory data analysis (figures are written to results/figures/)
jupyter notebook notebooks/01_EDA.ipynb

# 4) Threshold baseline (tables are written to results/)
python -m src.threshold_baseline
python -m src.threshold_baseline --window 5min --min-rules 1 --persistence 2 --horizon 6h
```

All random operations use the fixed seed in `src/config.py` (`RANDOM_SEED = 42`).

## 6. Milestones

| Date | Milestone | Deliverables | Status |
|---|---|---|---|
| 2026-10-20 | Proposal | One-page proposal, proposal slides (`docs/01_Proposal/`) | ✅ Done |
| 2026-11-03 | Literature map & method outline | Bibliography report, data download, EDA, threshold baseline | ✅ Done |
| 2026-11-24 | Methodology | Feature engineering, Isolation Forest, LSTM autoencoder | ⏳ Planned |
| 2026-12-08 | Methodology | Model tuning and refinement | ⏳ Planned |
| 2026-12-22 | Demo; experiments & analysis | Event-level evaluation, SHAP explanations | ⏳ Planned |
| 2027-01-12 | Final presentation & report | Final report, presentation, handout, demo video | ⏳ Planned |

## 7. Key references

1. Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021). Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry. In *2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA)*. IEEE. https://doi.org/10.1109/DSAA53316.2021.9564181
2. Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data, 9*, 764. https://doi.org/10.1038/s41597-022-01877-3
3. Jakobs, M., Veloso, B., & Gama, J. (2025). Interpretable rules for online failure prediction: A case study on the Metro do Porto dataset. *arXiv preprint* arXiv:2502.07394. https://arxiv.org/abs/2502.07394
4. Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R

## Licence

Code in this repository is provided for academic purposes within the MDM2 case study. The MetroPT-3 data remains under its original CC BY 4.0 licence and must be cited as above.
