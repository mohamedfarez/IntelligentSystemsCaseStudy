# Methodology Report, Phase 1 (Milestone 3)

Theoretical framework and system design for the case study *Early Warning of Air-Leak Failures in Metro Compressors: Explainable Anomaly Detection on the MetroPT-3 Dataset* (Group 6, submitted 24.11.2026).

**[Open the report (PDF)](Group6_Methodology_Phase1.pdf)**

| Section | Content |
|---|---|
| 1 Introduction | Context, previous milestones, objectives of this phase |
| 2 Theoretical Framework | Predictive maintenance as early failure warning, physics of an air leak, Isolation Forest, LSTM autoencoder, score-to-alarm logic, event-level evaluation, SHAP |
| 3 Materials and Methods | Research approach, workflow, data and inputs, preprocessing, algorithms and reasons for their selection, initial LSTM autoencoder design, alarm logic, evaluation protocol |
| 4 System Architecture | Five-layer design, modules and interfaces |
| 5 Initial Prototype | Verification and first results of the Isolation Forest against the threshold baseline |
| 6 Assumptions and Challenges | Main assumptions, limitations and implementation challenges |
| 7 Conclusion | Summary and next phase |

The prototype results come from `notebooks/02_preprocessing_iforest.ipynb`.

`source/` contains the LaTeX sources (`main.tex`, `abstract.tex`, `body.tex`, `references.bib`, `figures/`). Build with `latexmk -pdf main.tex`.
