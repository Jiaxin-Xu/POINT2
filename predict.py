import argparse
import pandas as pd
import numpy as np
import joblib

def predict_on_PI1M(model_path: str, pi1m_data_path: str) -> pd.DataFrame:
    """
    Predict polymer properties for the PI1M database using a trained model.
    
    Args:
        model_path (str): Path to the saved model file.
        pi1m_data_path (str): Path to the PI1M data CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the SMILES and predicted properties.
    """
    model = joblib.load(model_path)
    pi1m_data = pd.read_csv(pi1m_data_path)
    pi1m_smiles = pi1m_data['SMILES']
    X_pi1m = smiles_to_fingerprint(pi1m_smiles)

    pi1m_predictions = model.predict(X_pi1m)
    
    pi1m_data['Predicted_Tg'] = pi1m_predictions
    
    pi1m_data.to_csv('PI1M_Tg_predictions.csv', index=False)

    return pi1m_data

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for predicting polymer properties on the PI1M database.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Predict polymer properties on the PI1M database.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--pi1m_data_path', type=str, required=True, help='Path to the PI1M data CSV file.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    predict_on_PI1M(args.model_path, args.pi1m_data_path)
