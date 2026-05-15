# Quality-Control Decisions

## Data-Sharing Decision

Raw and processed EEG files are not included in this repository because the data belong to a supervised academic research project and may be subject to participant confidentiality and institutional data-sharing restrictions.

## Resting-State EEG QC

Resting-state EEG quality control was based on the percentage of clean 2-second epochs retained after amplitude-based artifact rejection.

One subject was excluded from resting-state analysis due to insufficient clean epochs.

## RDK Task EEG QC

Task EEG quality control was based on the percentage of task epochs retained after artifact rejection.

The following subjects were excluded from task-level analysis due to insufficient usable epochs:

```text
S14, S17, S19, S20

The final matched rest-task sample included:
S10, S11, S12, S13, S15, S18