# Methodology

## Resting-State EEG

Resting-state EEG was processed using MNE-Python. External BioSemi EXG channels were removed before analysis. EEG data were filtered from 1 to 40 Hz, notch-filtered at 50/100 Hz, resampled to 256 Hz, average referenced, and segmented into 2-second epochs.

High-amplitude epochs were rejected. Quality control was based on the percentage of clean epochs retained.

Resting-state spectral features were extracted using multitaper power spectral density estimation. Aperiodic parameters, including offset and exponent, were estimated using specparam.

## RDK Task EEG

Task EEG was processed using MNE-Python. RDK task triggers were identified from the Status channel.

The detected event codes were:

| Code | Condition |
|---:|---|
| 10 | interval1_random |
| 18 | interval1_coherent |
| 30 | interval2_random |
| 38 | interval2_coherent |

Task EEG was filtered from 0.1 to 40 Hz, notch-filtered, resampled to 256 Hz, average referenced, and epoched around RDK task events.

Task spectral features were extracted for posterior, occipital, and global ROIs. The main task contrast was coherent-minus-random spectral modulation.

## Rest–Task Integration

Resting-state EEG features were merged with task-related EEG modulation features at the subject level. The final matched sample included six participants.