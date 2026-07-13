import argparse
import numpy as np
from pathlib import Path
import h5py
import pandas as pd

def parse_trial_name(filename):
    """Extract trial type and condition from filename."""

    filename = Path(filename).stem

    # Remove the Segmented prefix
    name = filename.replace("Segmented_", "")

    parts = name.split("_")

    # Find the condition at the end of the filename
    condition = None

    for i in enumerate(parts):
            condition = parts[1]
            trial = parts[0]
            break

    if condition is None:
        raise ValueError(f"Could not determine condition from file: {filename}")

    return trial, condition


def open_trial(folder, trial, key=None):
    """ Open a single .mat trial file and determine the structure of the data. """

    file_path = Path(folder) / trial

    # Check if file exists, if not raise error
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # If file exists read it in .h5 format
    h5_file = h5py.File(file_path, 'r')

    # Find top level naming structure in .mat file, ignore "#ref#" errors
    top_keys = [k for k in h5_file.keys() if not k.startswith("#")]

    # If no no key is specified, check if there's only one key and use it
    # otherwise, prompt user to specify key
    if key is None:
        if len(top_keys) == 1:
            key = top_keys[0]
        else:
            raise ValueError(
                f"Multiple keys found: {top_keys}. Please specify the key containing trial data "
            )

    # Determine the sensor variables stored

    variables = [v for v in h5_file[key].keys() if not v.startswith("#")]

    
    records = []

    # Determine number of runs conducted based on the shape of each
    for variable in variables:
        runs = h5_file[key][variable].shape[0]

        # Store metadata - updated so that the extracted trial and condition
        # are also stored
        trial_name, condition = parse_trial_name(trial)

        records.append({
            "file": trial,
            "trial": trial_name,
            "condition": condition,
            "key": key,
            "variable": variable,
            "runs": runs
        })

    return pd.DataFrame(records)

def open_all_trials(folder, key=None):
    """Open all trials in 'data' folder and combine metadata"""

    folder = Path(folder)

    trials = sorted(folder.glob("*.mat"))

    if not trials:
        raise FileNotFoundError(f"No .mat files found in folder: {folder}")
    
    all_trials = []

    for trial in trials:
    
        print(f"Processing trial name {trial.name}")

        df = open_trial(folder, trial.name, key)

        all_trials.append(df)

    return pd.concat(all_trials, ignore_index=True)

def write_summary(df, output_file, output_folder="."):

    output_path = Path(output_folder) / output_file

    df.to_csv(output_path, index=False)

    print(f"Summary written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Open .mat files")

    parser.add_argument("--folder", default="data", help="Folder containing .mat trial data")

    parser.add_argument("--trial", help=".mat trial file name")

    parser.add_argument("--key", help="Top level key name containing trial data")

    parser.add_argument("--write", action="store_true", help="Write the trial metadata to a CSV file")

    args = parser.parse_args()

    # If trial is specified, only open that file
    if args.trial:
        df = open_trial(args.folder, args.trial, args.key)

        output_name = f"{Path(args.trial).stem}_summary.csv"

    # Otherwise open all .mat files in the data folder
    else:
        df = open_all_trials(args.folder, args.key)

        output_name = "all_trials_summary.csv"


    if args.write:
        write_summary(df, output_name)
    else:
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()