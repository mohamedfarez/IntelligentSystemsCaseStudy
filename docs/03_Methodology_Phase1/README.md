# Methodology Report, Phase 1 (Milestone 3)

Theoretical framework and system design for the case study *Early Warning of Air-Leak Failures in Metro Compressors: Explainable Anomaly Detection on the MetroPT-3 Dataset* (Group 6, submitted 24.11.2026).

**[Open the report (PDF)](Group6_Methodology_Phase1.pdf)** · [Presentation (PPTX)](Group6_Methodology_Phase1_Slides.pptx) · [Presentation (PDF)](Group6_Methodology_Phase1_Slides.pdf)

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

The presentation follows the pipeline as a metro line with seven stations (data, cleaning, windows, split, detectors, alarm, evaluation); every content slide pairs a figure with a technical layer, and speaker notes are included. Its figures were made from the real MetroPT-3 data with the phase-1 code; the pressure-cycle picture and the isolation and evaluation sketches are labelled as simulated or schematic. Transitions and animations need PowerPoint 2019 or Microsoft 365; the PDF shows the final state of every slide.

`source/` contains the LaTeX sources (`main.tex`, `abstract.tex`, `body.tex`, `references.bib`, `figures/`). Build with `latexmk -pdf main.tex`.
