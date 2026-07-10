import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

from open import open_trial


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
    trial_info = open_trial(folder, trial, key)

    for _, row in trial_info.iterrows():

        variable = row["variable"]
        key = row["key"]

        print(f"Processing {variable}")

        runs = extract_runs(
            folder,
            trial,
            key,
            variable
        )

        for run_number, signal in enumerate(runs, start=1):

            results.append({
                "trial": trial,
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
            "run": result["run"],
            "variable": result["variable"],
            "mean": np.mean(signal),
            "rms": np.sqrt(np.mean(signal**2)),
            "std": np.std(signal)
        })

    return pd.DataFrame(features)


def write_features(df, output_file="features.csv"):
    """Write extracted features to CSV."""

    df.to_csv(output_file, index=False)

    print(f"Features written to {output_file}")


def main():

    parser = argparse.ArgumentParser(description="Extract signal features")

    parser.add_argument("--folder", default="data")

    parser.add_argument("--trial")

    parser.add_argument("--key")

    parser.add_argument("--write", action="store_true")

    args = parser.parse_args()


    signals = extract_signal(
        args.folder,
        args.trial,
        args.key
    )

    df = extract_features(signals)


    if args.write:
        write_features(df)
    else:
        print(df)


if __name__ == "__main__":
    main()