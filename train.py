import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import joblib
from tpot import TPOTRegressor

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training and validating models.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train and validate models for polymer property prediction.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data.')
    parser.add_argument('--target_property', type=str, default='Tg', help='Target property to predict (e.g., Tg).')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of CPU cores to use.')
    return parser.parse_args()

def smiles_to_fingerprint(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Convert SMILES strings to numerical fingerprints using RDKit.
    
    Args:
        smiles_list (list[str]): List of SMILES strings.
        radius (int): Radius of the fingerprint.
        n_bits (int): Length of the fingerprint.

    Returns:
        np.ndarray: Array of numerical fingerprints.
    """
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprints.append(np.array(fingerprint))
    return np.array(fingerprints)

def train_validate(data_path: str, target_property: str = 'Tg', test_size: float = 0.2, n_jobs: int = -1) -> tuple:
    """
    Train and validate machine learning models for polymer property prediction.
    
    Args:
        data_path (str): Path to the CSV file containing the data.
        target_property (str): Target property to predict (e.g., Tg).
        test_size (float): Fraction of data to use for testing.
        n_jobs (int): Number of CPU cores to use.

    Returns:
        tuple: Trained model, training features, test features, training target, test target, and test predictions.
    """
    data = pd.read_csv(data_path)
    smiles = data['SMILES']
    y = data[target_property]

    X = smiles_to_fingerprint(smiles)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    automl = TPOTRegressor(verbosity=2, n_jobs=n_jobs)
    automl.fit(X_train, y_train)

    y_pred_train = automl.predict(X_train)
    y_pred_test = automl.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f'Train RMSE: {train_rmse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Test R2: {test_r2:.4f}')

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, label='Test Data')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Parity Plot')
    plt.legend()
    plt.show()

    automl.export('best_model.py')
    joblib.dump(automl.fitted_pipeline_, 'best_model.pkl')

    return automl, X_train, X_test, y_train, y_test, y_pred_test



if __name__ == "__main__":
    args = parse_args()
    train_validate(args.data_path, args.target_property, args.test_size, args.n_jobs)
