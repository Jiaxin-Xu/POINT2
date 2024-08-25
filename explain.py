import argparse
import shap
import joblib
import numpy as np

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model explanation.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Explain model predictions using SHAP.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--X_train_path', type=str, required=True, help='Path to the training features.')
    parser.add_argument('--X_test_path', type=str, required=True, help='Path to the test features.')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to the test target values.')
    return parser.parse_args()

def explain_model(model, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Generate SHAP values for model explanation.
    
    Args:
        model: Trained machine learning model.
        X_train (np.ndarray): Training feature data.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test target data.

    Returns:
        np.ndarray: SHAP values for the test data.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar")

    for i in range(3):
        shap.force_plot(explainer.expected_value, shap_values[i, :], X_test[i, :], matplotlib=True)

    return shap_values


if __name__ == "__main__":
    args = parse_args()
    model = joblib.load(args.model_path)
    X_train = np.load(args.X_train_path)
    X_test = np.load(args.X_test_path)
    y_test = np.load(args.y_test_path)
    explain_model(model, X_train, X_test, y_test)
