import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from pathlib import Path

from principal_Components import prepare_data
from principal_Components import compute_pca 

def train_SVM(results):
    """
    Create, train and evaluate Support Vector Machine (SVM)
    classifiers for each trial.

    Returns
    models - a dictionary containing trained SVM models for each trial.
    y_tests - test labels for the corresponding test features data
    y_preds - predicted labels for the corresponding test feature data
    """

    models = {}
    y_tests = {}
    y_preds = {}

    for trial_name in results["X_train_pca"]:

        print(f"\nTraining {trial_name}")

        X_train = results["X_train_pca"][trial_name]
        X_test = results["X_test_pca"][trial_name]

        y_train = results["y_train"][trial_name]
        y_test = results["y_test"][trial_name]

        # Create SVM model
        model = SVC(kernel="rbf")

        # Train model using PCA features
        model.fit(X_train, y_train)

        # Predict unseen test samples
        y_pred = model.predict(X_test)

        # Save trained model
        models[trial_name] = model
        y_tests[trial_name] = y_test
        y_preds[trial_name] = y_pred
    
    return(models, y_tests, y_preds)

def plot_confusion_Matrix(y_tests, y_preds):
    """
    Generate and save confusion matrices based on predicted and test labels 
    """
    print(y_tests.keys())
    for trial_name in y_tests:

        # Select folder to save confusion matrices
        save_dir = Path("plots")
        save_dir.mkdir(exist_ok=True) 

        # Display confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_tests[trial_name], y_preds[trial_name])

        plt.title(f"{trial_name} Confusion Matrix")
        plt.savefig(save_dir / f"{trial_name}-confusion_Matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

def main():
    # Load features and labels
    X, y = prepare_data("features.csv")

    # Scale and compute PCA
    results = compute_pca(X, y)

    # Execute SVM model
    models, y_tests, y_preds = train_SVM(results)

    # Plot and save confusion matrices
    plot_confusion_Matrix(y_tests, y_preds)

if __name__ == "__main__":
    main()
