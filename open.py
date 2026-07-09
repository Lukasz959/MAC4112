import argparse
import numpy as np
from pathlib import Path
import h5py
import pandas as pd
# Keys relating to sensor variables.
# All of the following variables are processed unless
# the variable is specified by the user


def open_trial(folder, trial, key=None):
    """ Open a .mat trial file and determine the structure of the data. """

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

    # Determine number of runs conducted based on the shape of the first variable
    
    records = []

    for variable in variables:
        runs = h5_file[key][variable].shape[0]

        records.append({
        "file": trial,
        "key": key,
        "variable": variable,
        "runs": runs
        })

    print(pd.DataFrame(records))
    return



def main():
    parser = argparse.ArgumentParser(description="Open .mat files")

    parser.add_argument("--folder", default="data", help="Folder containing .mat trial data")
    parser.add_argument("--trial", help=".mat trial file name")
    parser.add_argument("--key", help="Top level key name containing trial data")


    args = parser.parse_args()

    open_trial(args.folder, args.trial, args.key)

if __name__ == "__main__":
    main()