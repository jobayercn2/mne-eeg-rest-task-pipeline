from pathlib import Path
import re
import numpy as np
import mne


RAW_DIR = Path(r"I:\EEG_Python_Project\data\raw\task")


def get_subject_id(file_path):
    for part in reversed(file_path.parts):
        if re.match(r"^S\d+$", part, re.IGNORECASE):
            return part.upper()

    match = re.search(r"S\d+", file_path.stem, re.IGNORECASE)
    return match.group(0).upper() if match else file_path.stem


def load_raw_eeg(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".bdf":
        return mne.io.read_raw_bdf(file_path, preload=False, verbose="ERROR")
    elif suffix == ".edf":
        return mne.io.read_raw_edf(file_path, preload=False, verbose="ERROR")
    elif suffix == ".set":
        return mne.io.read_raw_eeglab(file_path, preload=False, verbose="ERROR")
    elif suffix == ".fif":
        return mne.io.read_raw_fif(file_path, preload=False, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def find_eeg_files(raw_dir):
    files = []
    for ext in ["*.bdf", "*.edf", "*.set", "*.fif"]:
        files.extend(raw_dir.rglob(ext))
    return sorted(files)


def inspect_file(file_path):
    subject = get_subject_id(file_path)

    print("\n" + "=" * 70)
    print(f"Subject: {subject}")
    print(f"File: {file_path}")
    print("=" * 70)

    raw = load_raw_eeg(file_path)

    print("\nChannels:")
    print(raw.ch_names)

    print("\nChannel types:")
    print(dict(zip(raw.ch_names, raw.get_channel_types())))

    # 1. Check annotations
    print("\nAnnotations:")
    if len(raw.annotations) > 0:
        print(raw.annotations)
        try:
            events_annot, event_id_annot = mne.events_from_annotations(raw, verbose="ERROR")
            print("\nAnnotation event_id:")
            print(event_id_annot)
            print("\nFirst 20 annotation events:")
            print(events_annot[:20])
        except Exception as e:
            print(f"Could not convert annotations to events: {e}")
    else:
        print("No annotations found.")

    # 2. Check likely stim/status channels
    stim_candidates = [
        ch for ch in raw.ch_names
        if ch.upper() in ["STATUS", "STI 014", "STI014", "TRIGGER", "TRIG"]
        or "STATUS" in ch.upper()
        or "TRIG" in ch.upper()
        or "STI" in ch.upper()
    ]

    print("\nPossible stim/status channels:")
    print(stim_candidates)

    for stim_ch in stim_candidates:
        print(f"\nTrying stim channel: {stim_ch}")
        try:
            events = mne.find_events(
                raw,
                stim_channel=stim_ch,
                shortest_event=1,
                verbose="ERROR"
            )

            if len(events) == 0:
                print("No events found.")
                continue

            codes, counts = np.unique(events[:, 2], return_counts=True)

            print("Detected event codes and counts:")
            for code, count in zip(codes, counts):
                print(f"  Code {int(code)}: {int(count)} events")

            print("\nFirst 30 events:")
            print(events[:30])

        except Exception as e:
            print(f"Could not read events from {stim_ch}: {e}")


def main():
    files = find_eeg_files(RAW_DIR)

    if not files:
        raise FileNotFoundError(f"No EEG files found in {RAW_DIR}")

    print(f"Found {len(files)} task EEG files.")

    # Inspect only first file first
    inspect_file(files[0])

    print("\nDone. First file inspected.")
    print("After you identify the codes, send me the event-code table.")


if __name__ == "__main__":
    main()