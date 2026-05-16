# MNE EEG Rest–Task Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MNE-Python](https://img.shields.io/badge/MNE--Python-EEG%20Analysis-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-purple)
![Data](https://img.shields.io/badge/Data-Not%20Public-lightgrey)

A reproducible computational EEG workflow for resting-state and random-dot kinematogram (RDK) visual motion task data using **MNE-Python**, **specparam**, spectral feature extraction, aperiodic parameterization, exploratory statistics, and a basic machine-learning workflow demonstration.

![Pipeline overview](figures/pipeline_overview.png)

---

## Project Snapshot

This repository demonstrates an end-to-end EEG analysis workflow:

- Resting-state EEG preprocessing and quality control
- RDK visual motion task event inspection and epoching
- Spectral band-power feature extraction
- Aperiodic EEG parameterization using `specparam`
- Posterior ROI-level rest–task feature integration
- Exploratory correlation analysis
- Basic leave-one-subject-out machine-learning workflow demonstration

Raw and processed EEG data are not publicly shared due to research-data ownership and participant confidentiality restrictions.

---

## Overview

This repository implements a complete EEG analysis pipeline in Python using MNE-Python. The workflow processes resting-state EEG and random-dot kinematogram (RDK) task EEG data, performs quality control, extracts spectral and aperiodic EEG features, merges rest–task features, and runs exploratory statistical and machine-learning analyses.

The project is designed as a computational neuroscience portfolio project demonstrating EEG preprocessing, quality control, event handling, spectral feature extraction, aperiodic parameterization, rest–task integration, and reproducible scientific workflow design.

---

## Research Aim

The main aim of this project is to examine whether resting-state EEG features are related to task-related EEG modulation during visual motion processing.

The central exploratory question is:

> Are resting-state spectral and aperiodic EEG features associated with coherent-versus-random RDK task EEG modulation?

---

## Why This Project Matters

This project was designed to demonstrate reproducible computational neuroscience workflow construction rather than confirmatory statistical inference.

The pipeline shows how resting-state EEG features can be extracted, quality-controlled, integrated with task-related EEG modulation, and analyzed using transparent Python scripts. The emphasis is on responsible data handling, interpretable EEG features, and reproducible research organization.

---

## Data Availability

Raw and processed EEG data are **not included** in this repository.

The EEG recordings belong to a supervised academic research project and may be subject to institutional data-sharing restrictions and participant confidentiality requirements.

This repository therefore focuses on the computational workflow, code organization, feature extraction strategy, quality-control decisions, and selected summary outputs.

---

## Example Outputs

### Pipeline Overview

![Pipeline overview](figures/pipeline_overview.png)

### Example Resting-State PSD

![Example PSD](results/figures/example_psd.png)

### Rest–Task Correlation Heatmap

![Correlation heatmap](results/figures/correlation_heatmap.png)

### Example Exploratory Association

![Rest exponent vs task alpha modulation](results/figures/rest_exponent_vs_task_alpha_power_diff.png)

### ML Demonstration Output

![Ridge regression predicted vs observed](results/figures/ridge_regression_task_alpha_power_diff_predicted_vs_observed.png)

---

## Repository Structure

```text
mne-eeg-rest-task-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── 01_preprocess_resting.py
│   ├── 02_qc_resting.py
│   ├── 03_extract_resting_features.py
│   ├── 04_inspect_rdk_events.py
│   ├── 05_preprocess_rdk_task.py
│   ├── 06_extract_task_features.py
│   ├── 07_merge_rest_task.py
│   ├── 08_exploratory_analysis.py
│   ├── 09_ml_demo_loocv.py
│   └── 10_make_pipeline_overview.py
│
├── docs/
│   ├── project_summary.md
│   ├── methodology.md
│   ├── qc_decisions.md
│   └── interpretation_notes.md
│
├── results/
│   ├── tables/
│   └── figures/
│
└── figures/
    └── pipeline_overview.png
```

---

## Expected Local Data Structure

Raw and processed EEG data are **not included** in this repository. To reproduce the pipeline with authorized data access, organize files locally as:

```text
data/
├── raw/
│   ├── resting_state/
│   │   ├── S10/
│   │   ├── S11/
│   │   └── ...
│   │
│   └── task/
│       ├── S10/
│       ├── S11/
│       └── ...
│
└── processed/
    ├── resting_state/
    └── task/
```

The `data/` folder is excluded from GitHub using `.gitignore`.

---

## Software and Libraries

The pipeline uses:

- Python
- MNE-Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- specparam

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Pipeline Summary

The complete workflow is:

```text
Raw EEG
→ preprocessing
→ quality control
→ feature extraction
→ rest–task feature merge
→ exploratory statistics
→ machine-learning workflow demonstration
```

---

## 1. Resting-State EEG Preprocessing

Script:

```text
scripts/01_preprocess_resting.py
```

Main steps:

1. Load raw resting-state EEG files.
2. Remove BioSemi EXG channels.
3. Apply standard 10-20 montage.
4. Band-pass filter from 1 to 40 Hz.
5. Apply 50/100 Hz notch filtering.
6. Resample to 256 Hz.
7. Apply average reference.
8. Create 2-second fixed-length epochs.
9. Reject high-amplitude noisy epochs.
10. Save cleaned raw and epoched EEG files locally.

---

## 2. Resting-State Quality Control

Script:

```text
scripts/02_qc_resting.py
```

The QC procedure summarizes:

- Number of epochs retained
- Percentage of epochs retained
- Subject-level QC flags
- PSD plots for visual inspection

One subject was excluded from resting-state analysis due to insufficient clean epochs.

---

## 3. Resting-State Feature Extraction

Script:

```text
scripts/03_extract_resting_features.py
```

Extracted resting-state features include:

- Delta power
- Theta power
- Alpha power
- Beta power
- Gamma power
- Relative band power
- Aperiodic offset
- Aperiodic exponent
- Alpha peak frequency
- Alpha peak power
- Model fit quality metrics

Aperiodic EEG features were estimated using `specparam`.

Primary resting-state ROI:

```text
posterior
```

Posterior ROI channels:

```text
O1, O2, Oz, PO3, PO4, PO7, PO8
```

---

## 4. RDK Task Event Inspection

Script:

```text
scripts/04_inspect_rdk_events.py
```

This script inspects the task EEG Status channel and identifies RDK event codes.

Detected task event codes:

| Event Code | Condition |
|---:|---|
| 10 | interval1_random |
| 18 | interval1_coherent |
| 30 | interval2_random |
| 38 | interval2_coherent |

---

## 5. RDK Task EEG Preprocessing

Script:

```text
scripts/05_preprocess_rdk_task.py
```

Main steps:

1. Load raw RDK task EEG data.
2. Drop BioSemi EXG channels.
3. Apply standard montage.
4. Filter EEG from 0.1 to 40 Hz.
5. Apply 50/100 Hz notch filtering.
6. Resample to 256 Hz.
7. Apply average reference.
8. Epoch around RDK event codes.
9. Apply baseline correction.
10. Reject noisy epochs.
11. Save cleaned task epochs locally.

---

## 6. RDK Task Feature Extraction

Script:

```text
scripts/06_extract_task_features.py
```

The task feature extraction script computes spectral features for each event condition and ROI.

Task ROIs:

- Posterior
- Occipital
- Global

The main task contrast is:

```text
coherent_minus_random
```

This contrast represents task-related spectral modulation between coherent and random motion conditions.

---

## 7. Rest–Task Feature Merge

Script:

```text
scripts/07_merge_rest_task.py
```

This script merges resting-state EEG features with RDK task EEG modulation features at the subject level.

Final matched sample:

```text
N = 6
```

Final included subjects:

```text
S10, S11, S12, S13, S15, S18
```

---

## 8. Exploratory Statistical Analysis

Script:

```text
scripts/08_exploratory_analysis.py
```

This script produces:

- Descriptive statistics
- Pearson correlations
- Spearman correlations
- Scatterplots
- Correlation heatmap

Main resting-state predictors:

```text
rest_exponent
rest_offset
rest_alpha_power
rest_alpha_relative_power
rest_theta_power
rest_beta_power
```

Main task outcomes:

```text
task_theta_power_diff
task_alpha_power_diff
task_beta_power_diff
task_gamma_power_diff
```

---

## 9. Machine-Learning Demonstration

Script:

```text
scripts/09_ml_demo_loocv.py
```

A proof-of-concept machine-learning workflow was implemented using resting-state EEG features to predict RDK task-related spectral modulation.

Models used:

- Linear regression
- Ridge regression
- Random forest regression

Validation approach:

```text
leave-one-subject-out cross-validation
```

Because the final matched sample size is small, this analysis is interpreted strictly as a workflow demonstration rather than a validated predictive model.

---

## Key Exploratory Outputs

After quality control, the final matched rest–task sample included six participants. Therefore, all statistical and machine-learning outputs are interpreted as exploratory and workflow-demonstration level.

The pipeline produced:

- Posterior ROI resting-state spectral and aperiodic features
- RDK task coherent-minus-random spectral modulation features
- Merged rest–task feature tables
- Pearson and Spearman correlation tables
- Scatterplots and a correlation heatmap
- A leave-one-subject-out regression-based ML demonstration

Exploratory analyses suggested that resting-state spectral features may relate to task-related gamma-band modulation. These results are treated as preliminary and hypothesis-generating rather than confirmatory.

---

## Quality-Control Decisions

### Resting-State EEG

Resting-state EEG was preprocessed and segmented into 2-second epochs. Subject-level QC was based on the percentage of clean epochs retained after artifact rejection.

One participant was excluded from resting-state analysis due to insufficient clean epochs.

### RDK Task EEG

The following participants were excluded from task-level analysis due to insufficient clean epochs:

```text
S14, S17, S19, S20
```

The final matched rest–task sample included:

```text
S10, S11, S12, S13, S15, S18
```

---

## Machine-Learning Results

The machine-learning extension showed limited predictive performance, with mostly negative cross-validated R² values. This was expected given the final matched sample size.

The ML component is included to demonstrate:

- Feature-table construction
- Train/test workflow
- Leave-one-subject-out validation
- Regression-based prediction
- Model evaluation
- Coefficient inspection

It is not presented as a validated predictive model.

---

## How to Run the Pipeline

After placing authorized EEG data locally in the expected structure, run:

```bash
python scripts/01_preprocess_resting.py
python scripts/02_qc_resting.py
python scripts/03_extract_resting_features.py
python scripts/04_inspect_rdk_events.py
python scripts/05_preprocess_rdk_task.py
python scripts/06_extract_task_features.py
python scripts/07_merge_rest_task.py
python scripts/08_exploratory_analysis.py
python scripts/09_ml_demo_loocv.py
```

To regenerate the pipeline overview figure:

```bash
python scripts/10_make_pipeline_overview.py
```

---

## Skills Demonstrated

This project demonstrates:

- EEG preprocessing with MNE-Python
- Resting-state EEG analysis
- RDK task EEG analysis
- Event-code inspection and event-based epoching
- Spectral feature extraction
- Aperiodic EEG parameterization with specparam
- ROI-level EEG analysis
- Quality-control workflow
- Rest–task feature integration
- Exploratory statistical analysis
- Basic machine-learning workflow with scikit-learn
- Reproducible computational neuroscience project organization

---

## Limitations

This project is exploratory and pipeline-focused.

Main limitations:

1. Small final matched sample size.
2. Raw and processed EEG data are not included.
3. Task EEG quality varied across participants.
4. Machine-learning results are demonstration-level only.
5. Statistical findings should not be interpreted as confirmatory evidence.

---

## Disclaimer

This repository is intended as a computational neuroscience portfolio and methodological demonstration project. Due to data ownership, participant confidentiality restrictions, and the small final sample size, the repository emphasizes reproducible workflow design rather than confirmatory scientific inference.
```
