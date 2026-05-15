# EEG Rest–Task Analysis Pipeline

A reproducible computational EEG analysis pipeline for resting-state and RDK visual motion task data using MNE-Python, spectral feature extraction, aperiodic parameterization, exploratory statistics, and a machine-learning workflow demonstration.

## Overview

This repository implements a complete EEG analysis pipeline in Python using MNE-Python. The workflow processes resting-state EEG and random-dot kinematogram (RDK) task EEG data, performs quality control, extracts spectral and aperiodic EEG features, merges rest-task features, and runs exploratory statistical and machine-learning analyses.

The project is designed as a computational neuroscience portfolio project demonstrating EEG preprocessing, quality control, event handling, spectral feature extraction, aperiodic parameterization, rest-task integration, and reproducible scientific workflow design.

## Research Aim

The main aim of this project is to examine whether resting-state EEG features are related to task-related EEG modulation during visual motion processing.

The central exploratory question is:

> Are resting-state spectral and aperiodic EEG features associated with coherent-versus-random RDK task EEG modulation?

## Data Availability

Raw and processed EEG data are not included in this repository because the recordings belong to a supervised academic research project and are subject to data-sharing and participant confidentiality restrictions.

This repository is intended to demonstrate the computational workflow, preprocessing pipeline, feature extraction methods, quality-control strategy, exploratory analysis, and project organization.

To reproduce the workflow with appropriate permissions, raw data should be placed locally following the expected data structure described below.

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
│   └── 09_ml_demo_loocv.py
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
