import pandas as pd
import numpy as np
import joblib
from torch_molecule import GREAMolecularPredictor, GNNMolecularPredictor
import os
from train import smiles_to_fingerprint
from model import mlp_predict_with_uncertainty
import torch

def predict_on_unlabeled(target_property, model_fpmethod_pairs, unlabeled_data_path):
    """
    Predict polymer properties for the PI1M database using paired models and fingerprint methods.
    
    Args:
        target_property (str): Target property to predict.
        model_fpmethod_pairs (list): List of paired models and fingerprint methods with radius and n_bits.
        unlabeled_data_path (str): Path to the unlabeled (e.g., PI1M) data CSV file.
    
    Returns:
        None
    """
    unlabeled_data = pd.read_csv(unlabeled_data_path)
    
    # # Randomly sample a subset of the data (e.g., 20% of the total data) -- model memory issue
    # subset_fraction = 0.2  # Change this value as needed
    # unlabeled_data = unlabeled_data.sample(frac=subset_fraction, random_state=42)

    unlabeled_smiles = unlabeled_data['SMILES']

    output_dir = "screen"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{target_property}_predictions.csv")

    prediction_results = unlabeled_data[['SMILES']].copy()

    for pair in model_fpmethod_pairs:
        model_name, fpmethod, radius, n_bits = pair
        model_path = f"./results/{target_property}_uq/{model_name}_{fpmethod}_{radius}_{n_bits}/best_model_with_uq.pkl"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        print(f"Loading model: {model_path}")
        if model_name.startswith('torch-G'):  # Check for graph-based models
            if 'GREA' in model_name:
                model = GREAMolecularPredictor()
                model.load_model(model_path)
                # try:
                #     model.load_model(model_path, map_location=torch.device('cpu'))
                # except ValueError as e:
                #     print(f"Error loading model: {e}")
                #     continue
                eval_results = model.predict(unlabeled_smiles.tolist())
                y_mean = eval_results['prediction'].reshape(-1)
                variance = eval_results['variance'].reshape(-1)
                confidence_multiplier = 1.96  # For 95% confidence interval
                lower_quantile = y_mean - confidence_multiplier * np.sqrt(variance)
                upper_quantile = y_mean + confidence_multiplier * np.sqrt(variance)
                std_dev = (upper_quantile - lower_quantile) / 2
                predictions = np.column_stack((y_mean, std_dev))
            else:
                model = GNNMolecularPredictor()
                model.load_model(model_path)
                # try:
                #     model.load_model(model_path, map_location=torch.device('cpu'))
                # except ValueError as e:
                #     print(f"Error loading model: {e}")
                #     continue
                predictions = model.predict(unlabeled_smiles.tolist())['prediction']
        else:
            model = joblib.load(model_path)
            X_pi1m = smiles_to_fingerprint(unlabeled_smiles, fpmethod, radius, n_bits)
            if model_name == 'QuantileRandomForest':
                y_mean = model.predict(X_pi1m)
                lower_quantile = model.predict(X_pi1m, quantiles=0.05)
                upper_quantile = model.predict(X_pi1m, quantiles=0.95)
                std_dev = (upper_quantile - lower_quantile) / 2
                predictions = np.column_stack((y_mean, std_dev))
            elif model_name == 'DropoutMLP':
                y_mean, _, quantile_predictions = mlp_predict_with_uncertainty(
                    model, X_pi1m, n_iter=100, quantiles=[5, 95])
                lower_quantile = quantile_predictions[0]
                upper_quantile = quantile_predictions[1]
                std_dev = (upper_quantile - lower_quantile) / 2
                predictions = np.column_stack((y_mean, std_dev))
            else:
                predictions = model.predict(X_pi1m)
        prediction_column = f"{model_name}_{fpmethod}_predicted_{target_property}"
        # prediction_results[prediction_column] = predictions
        prediction_results[f"{model_name}_{fpmethod}_{target_property}_mean"] = predictions[:, 0]  # Mean predictions
        prediction_results[f"{model_name}_{fpmethod}_{target_property}_std"] = predictions[:, 1]  # Standard deviation


    prediction_results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # List of target properties
    target_properties = [
        # "Tg", 
        # "Tm", "TC", "FFV", "density",
        # "O2_msa_log10", "N2_msa_log10", "CO2_msa_log10", "H2_msa_log10", "CH4_msa_log10",
        "CH4_log10_ladder_fromlinear_ladder", "CO2_log10_ladder_fromlinear_ladder", 
        "N2_log10_ladder_fromlinear_ladder", "O2_log10_ladder_fromlinear_ladder", 
        "H2_log10_ladder_fromlinear_ladder"

    ]

    # List of model-fpmethod pairs with radius and n_bits
    model_fpmethod_pairs = [
        # ("torch-GREA", "Morgan", 2, 2048),
        # ("torch-GNN", "Morgan", 2, 2048),
        ("QuantileRandomForest", "Morgan", 2, 2048),
        ("QuantileRandomForest", "MACCS", 2, 2048),
        ("QuantileRandomForest", "AtomPair", 2, 2048),
        ("QuantileRandomForest", "TopologicalTorsion", 2, 2048),
        ("QuantileRandomForest", "RDKit", 6, 2048),
        ("DropoutMLP", "Morgan", 2, 2048),
        ("DropoutMLP", "MACCS", 2, 2048),
        ("DropoutMLP", "AtomPair", 2, 2048),
        ("DropoutMLP", "TopologicalTorsion", 2, 2048),
        ("DropoutMLP", "RDKit", 6, 2048)
    ]

    # unlabeled_data_path = "./data/unlabeled/PI1M/PI1M_v2.csv"  # Path to the unlabeled data CSV
    unlabeled_data_path = "./data/unlabeled/ladder_project/rule_based_allcombined_unique.csv"  # Path to the unlabeled data CSV

    for target_property in target_properties:
        print(f"Starting predictions for {target_property}...")
        predict_on_unlabeled(
            target_property=target_property,
            model_fpmethod_pairs=model_fpmethod_pairs,
            unlabeled_data_path=unlabeled_data_path
        )
        print(f"Finished predictions for {target_property}.")
