import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import h5py
from scipy.stats import kurtosis, skew

from open import open_trial 
from open import open_all_trials

# Variable names relating to sensor measurements.
# Used laterto filter out machine tool PLC data before processing
SENSOR_VARIABLES = {
    "SpindleAccX", "SpindleAccY", "SpindleAccZ",
    "PlateLFAccX", "PlateLFAccY", "PlateLFAccZ",
    "PlateHFAccZ", "Power"
}

# Account for the difference in naming style for the Machining trial
VARIABLE_ALIASES = {
    "SpindleX": "SpindleAccX",
    "SpindleY": "SpindleAccY",
    "SpindleZ": "SpindleAccZ",
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
        raw_variable = row["variable"]  # Keep original name for HDF5 lookup
        key = row["key"]

        # Convert to standard variable name for filtering and downstream features
        standard_variable = VARIABLE_ALIASES.get(raw_variable, raw_variable)

        # Print when moving to a new trial
        if trial != current_trial:
            print(f"\nProcessing trial: {trial}-{condition}")
            current_trial = trial

        # Filter using the standardized name
        if standard_variable not in SENSOR_VARIABLES:
            print(f"Skipping {raw_variable}")
            continue

        print(f"Processing {raw_variable}")

        # Pass raw_variable so h5py can look up the exact key in the file
        runs = extract_runs(
            folder,
            filename,
            key,
            raw_variable
        )

        for run_number, signal in enumerate(runs, start=1):

            # Append using standard_variable so your final CSV features are uniform
            results.append({
                "trial": trial,
                "condition": condition,
                "run": run_number,
                "variable": standard_variable,
                "signal": signal
            })

    return results


def extract_features(results):
    """Calculate statistical features for each run and create DataFrame."""

    rows = {}

    for result in results:

        # Unique identifier for each run
        key = (
            result["trial"],
            result["condition"],
            result["run"]
        )

        # Create a new row if this run hasn't been seen before
        if key not in rows:
            rows[key] = {
                "trial": result["trial"],
                "condition": result["condition"],
                "run": result["run"]
            }

        signal = result["signal"]
        variable = result["variable"]

        # Intermediate calculations
        abs_signal = np.abs(signal)
        mean_abs = np.mean(abs_signal)
        max_abs = np.max(abs_signal)
        rms = np.sqrt(np.mean(signal**2))
        mean_sqrt_abs = np.mean(np.sqrt(abs_signal))
        
        # Avoid division by zero in calculation of factors
        e = 1e-9
        mean_abs_safe = mean_abs if mean_abs > 0 else e
        rms_safe = rms if rms > 0 else e
        mean_sqrt_abs_safe = mean_sqrt_abs if mean_sqrt_abs > 0 else e

        # Add features for this variable
        rows[key][f"{variable}_mean"] = np.mean(signal)
        rows[key][f"{variable}_rms"] = rms
        rows[key][f"{variable}_std"] = np.std(signal)
        rows[key][f"{variable}_kurtosis"] = kurtosis(signal)
        rows[key][f"{variable}_skewness"] = skew(signal)
        rows[key][f"{variable}_peak_to_peak"] = np.ptp(signal)
        rows[key][f"{variable}_energy"] = np.mean(signal**2)

        # Add factors
        rows[key][f"{variable}_crest_factor"] = max_abs / rms_safe
        rows[key][f"{variable}_shape_factor"] = rms / mean_abs_safe
        rows[key][f"{variable}_impulse_factor"] = max_abs / mean_abs_safe
        rows[key][f"{variable}_margin_factor"] = max_abs / (mean_sqrt_abs_safe**2)
    return pd.DataFrame(rows.values())

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

    start_time = time.perf_counter()

    signals = extract_signal(args.folder, args.trial, args.key)

    df = extract_features(signals)

    if args.write:
        write_features(df)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()