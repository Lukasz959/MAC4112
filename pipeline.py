import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

from open import open_trial 
from open import open_all_trials

# Variable names relating to sensor measurements.
# Used laterto filter out machine tool PLC data before processing
SENSOR_VARIABLES = {
    "SpindleAccX", "SpindleAccY", "SpindleAccZ",
    "PlateLFAccX", "PlateLFAccY", "PlateLFAccZ",
    "PlateHFAccZ", "Power"
}

def extract_runs(folder, trial, key, variable):
    """Extract all runs for a specific variable."""

    file_path = Path(folder) / trial

    runs = []

    with h5py.File(file_path, "r") as h5_file:

        references = h5_file[key][variable]

        for ref in references[:, 0]:

            run = h5_file[ref][:]

            runs.append(run.flatten())

    return runs


def extract_signal(folder, trial, key=None):
    """Extract signal for every run of every variable."""

    results = []

    # Get file structure
    # If trial is specified process that trial, otherwise process all
    if trial is None:
        trial_info = open_all_trials(folder, key)
    else:
        trial_info = open_trial(folder, trial, key)

    current_trial = None

    for _, row in trial_info.iterrows():

        filename = row["file"]
        trial = row["trial"]
        condition = row["condition"]
        variable = row["variable"]
        key = row["key"]

        # Print when moving to a new trial
        if trial != current_trial:
            print(f"\nProcessing trial: {trial}")
            current_trial = trial

        # Skip non-sensor variables
        if variable not in SENSOR_VARIABLES:
            print(f"Skipping {variable}")
            continue

        print(f"Processing {variable}")

        runs = extract_runs(
            folder,
            filename,
            key,
            variable
        )

        for run_number, signal in enumerate(runs, start=1):

            results.append({
                "trial": trial,
                "condition": condition,
                "run": run_number,
                "variable": variable,
                "signal": signal
            })

    return results


def extract_features(results):
    """Calculate statistical features for each signal and create DataFrame."""

    features = []

    for result in results:

        signal = result["signal"]

        features.append({
            "trial": result["trial"],
            "condition": result["condition"],
            "run": result["run"],
            "variable": result["variable"],
            "mean": np.mean(signal),
            "rms": np.sqrt(np.mean(signal**2)),
            "std": np.std(signal)
        })

    return pd.DataFrame(features)

def write_features(df, output_file="features.csv"):
    """Write extracted features to CSV."""

    print(f"Writing features to {output_file}")

    df.to_csv(output_file, index=False)

    print(f"Features written to {output_file}")


def main():

    parser = argparse.ArgumentParser(description="Extract signal features")

    parser.add_argument("--folder", default="data")

    parser.add_argument("--trial")

    parser.add_argument("--key")

    parser.add_argument("--write", action="store_true")

    args = parser.parse_args()

    signals = extract_signal(args.folder, args.trial, args.key)

    df = extract_features(signals)

    if args.write:
        write_features(df)

if __name__ == "__main__":
    main()