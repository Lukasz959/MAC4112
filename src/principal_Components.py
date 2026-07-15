import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

def prepare_data(filepath):
    """
    Load extracted signal features from a CSV file and organise them by trial.

    The first column of the CSV is assumed to contain the trial name.
    The second column contains the target labels.
    Columns from index 3 onwards are treated as input features.

    Returns: 
    X - a dictionary contining feature data frames for each trial
    y - a dictionary containing labels for each trial.
    """

    df = pd.read_csv(filepath)

    X = {}
    y = {}

    # Separate the data set into individual trials
    for trial_name, trial_df in df.groupby(df.columns[0]):

        X_trial = trial_df.iloc[:, 3:].copy()
        y_trial = trial_df.iloc[:, 1].copy()

        # Remove feature columns containing NaN values
        # HF accelerometer column is expected to be removed from the machining trial
        # since there is no data for this sensor in this trial
        deleted_columns = X_trial.columns[X_trial.isna().any()]

        # Print the removed column header in case other unexpected columns are removed 
        if len(deleted_columns) > 0:
            print(f"{trial_name}: Removing {list(deleted_columns)}")

        X_trial = X_trial.drop(columns=deleted_columns)

        X[trial_name] = X_trial
        y[trial_name] = y_trial

    return X, y

def compute_pca(X, y):
    """
    Applies PCA to feature data following the steps:
    1. Splits data into training and test datasets
    2. Stadardises features using training data statistics
    3. Computes PCA on training data only
    4. Converts testing and training features into the principal component space

    Returns:
    Results - a dictionary containing:
        - X_train: Standardised training features
        - X_test: Standardised testing features
        - y_train: Training labels
        - y_test: Testing labels
        - X_train_pca: PCA-transformed training features
        - X_test_pca: PCA-transformed testing features
        - scalers: StandardScaler objects for each trial
        - pcas: PCA objects for each trial
    """

    results = {
        "X_train": {},
        "X_test": {},
        "y_train": {},
        "y_test": {},
        "X_train_pca": {},
        "X_test_pca": {},
        "scalers": {},
        "pcas": {}
    }

    for trial_name, X_trial in X.items():

        print(f"Processing {trial_name}")

        # Split data 65:35 (training:test)
        (
            results["X_train"][trial_name],
            results["X_test"][trial_name],
            results["y_train"][trial_name],
            results["y_test"][trial_name]
        ) = train_test_split(
            X_trial,
            y[trial_name],
            test_size=0.35,
            random_state=21
        )

        # Apply scaling to standardise features 
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(results["X_train"][trial_name])

        # Apply the same scaling parameters to unseen test data
        X_test_scaled = scaler.transform(results["X_test"][trial_name])

        # Store scalers
        results["scalers"][trial_name] = scaler

        # Update results dictionary with new scaled variables
        results["X_train"][trial_name] = pd.DataFrame(X_train_scaled,columns=X_trial.columns)
        results["X_test"][trial_name] = pd.DataFrame(X_test_scaled,columns=X_trial.columns)

        # Compute 3 component PCA
        pca = PCA(n_components=3)

        # Fit PCA only using training data
        results["X_train_pca"][trial_name] = pca.fit_transform(results["X_train"][trial_name])

        # Convert test data into the same PCA coordinate system
        results["X_test_pca"][trial_name] = pca.transform(results["X_test"][trial_name])

        results["pcas"][trial_name] = pca

    return results

def plot_pca(X_pca, y_train, pcas):
    """
    Generate and save 3D PCA scatter plots for each trial
    Scatter plots are saved in the 'plots' folder.
    """

    save_dir = Path("../plots")
    save_dir.mkdir(exist_ok=True)

    for trial_name, X_trial_pca in X_pca.items():

        print(f"Plotting {trial_name}")

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")

        y_trial = y_train[trial_name]

        # Plot each condition/label using a different marker colour
        for condition in np.unique(y_trial):

            mask = y_trial == condition

            ax.scatter(X_trial_pca[mask, 0], X_trial_pca[mask, 1], X_trial_pca[mask, 2], label=condition, alpha=0.6)

        pca_model = pcas[trial_name]

        # Add axis titles with percentage explained varaince
        ax.set_xlabel(f"PCA1 {pca_model.explained_variance_ratio_[0]*100:.1f}%")
        ax.set_ylabel(f"PCA2 {pca_model.explained_variance_ratio_[1]*100:.1f}%")
        ax.set_zlabel(f"PCA3 {pca_model.explained_variance_ratio_[2]*100:.1f}%")

        ax.set_title(f"PCA - {trial_name}")
        ax.legend()

        plt.savefig(save_dir / f"{trial_name}-PCA.png", dpi=300, bbox_inches="tight")

        plt.close(fig)

def main():

    X, y = prepare_data("../features.csv")

    results = compute_pca(X, y)

    plot_pca(results["X_train_pca"], results["y_train"], results["pcas"])

if __name__ == "__main__":
    main()