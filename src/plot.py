import argparse
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def read_features(feature_file):
    """Read the extracted features."""

    return pd.read_csv(feature_file)

def plot_features(df, trial, variable, x, y, z):
    """Create a 3D scatter plot."""

    # Filter the data
    df = df[
        (df["trial"] == trial) &
        (df["variable"] == variable)
    ]

    if df.empty:
        raise ValueError(
            f"No data found for trial '{trial}' and variable '{variable}'"
        )

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    # Plot each condition separately
    for condition in sorted(df["condition"].unique()):

        subset = df[df["condition"] == condition]

        ax.scatter(
            subset[x],
            subset[y],
            subset[z],
            label=condition
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    ax.set_title(f"{trial} - {variable}")

    ax.legend()

    plt.show()

def main():

    parser = argparse.ArgumentParser(
        description="Plot extracted signal features"
    )

    parser.add_argument("--features", default="features.csv", help="Feature CSV file")

    parser.add_argument("--trial", required=True, help="Trial to plot")

    parser.add_argument("--variable", required=True, help="Variable to plot")

    parser.add_argument("--x", default="mean")

    parser.add_argument("--y", default="rms")

    parser.add_argument("--z", default="std")

    args = parser.parse_args()

    df = read_features(args.features)

    plot_features(
        df,
        args.trial,
        args.variable,
        args.x,
        args.y,
        args.z
    )


if __name__ == "__main__":
    main()