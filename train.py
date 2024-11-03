import argparse
import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, MACCSkeys, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import joblib
from tpot import TPOTRegressor
import warnings 
from quantile_forest import RandomForestQuantileRegressor
from utils import set_global_random_seed
import tensorflow as tf

from model import train_mlp_model, mlp_predict_with_uncertainty, build_bayesian_model, compile_and_train_BNN, bnn_predict_with_uncertainty
    
def setup_logging(results_dir: str):
    """
    Redirect stdout and stderr to a log file in the results directory.

    Args:
        results_dir (str): Path to the directory where the log file will be saved.
    """
    log_file = os.path.join(results_dir, 'run.log')
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training and validating models.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train and validate models for polymer property prediction.')
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data.')
    parser.add_argument('--fpmethod', type=str, default='Morgan', choices=['Morgan', 'RDKit', 'MACCS', 'TopologicalTorsion', 'AtomPair'], help='Fingerprinting method.')
    parser.add_argument('--radius', type=int, default=2, help='Radius for Morgan and Topological Torsion.')
    parser.add_argument('--n_bits', type=int, default=2048, help='Number of bits for the fingerprint vector.')

    parser.add_argument('--target_property', type=str, default='Tg', help='Target property to predict (e.g., Tg).')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of CPU cores to use.')

    parser.add_argument('--model', type=str, default='QuantileRandomForest', choices=['QuantileRandomForest', 'DropoutMLP', 'BNN'], help='Uncertainty Model Type.')
    return parser.parse_args()

# def smiles_to_fingerprint(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
#     """
#     Convert SMILES strings to numerical fingerprints using RDKit's MorganGenerator.
#     More info on creating fingerprints using rdkit: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    
#     Args:
#         smiles_list (list[str]): List of SMILES strings.
#         radius (int): Radius of the fingerprint.
#         n_bits (int): Length of the fingerprint.

#     Returns:
#         np.ndarray: Array of numerical fingerprints.
#     """
#     morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
#     fingerprints = []
#     for smiles in smiles_list:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:  # Ensure valid molecule
#             fingerprint = morgan_generator.GetFingerprint(mol)
#             fingerprints.append(np.array(fingerprint))
#     return np.array(fingerprints)

def smiles_to_fingerprint(smiles_list: list[str], method: str = 'Morgan', 
                          radius: int = 2, n_bits: int = 2048, 
                          return_bit_info=False) -> np.ndarray:
    """
    Convert SMILES strings to numerical fingerprints based on the specified method using RDKit.
    Optionally returns bit information for Morgan fingerprints.
    
    Args:
        smiles_list (list[str]): List of SMILES strings.
        method (str): Method of fingerprinting. Supported methods: 'Morgan', 'RDKit', 'MACCS', 'TopologicalTorsion', 'AtomPair'.
        radius (int): Radius of the fingerprint (applicable for Morgan and TopologicalTorsion).
        n_bits (int): Length of the fingerprint (applicable for Morgan, RDKit, and AtomPair).

    Returns:
        np.ndarray: Array of numerical fingerprints.
    """
    fingerprints = []
    all_bit_info = []  # Store bit information for all molecules
    
    # ignore the molecule, just save all the bits in the dataset in one
    list_bits = []
    legends = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:  # Ensure valid molecule
            continue

        if method == 'Morgan':
            bitInfo = {}
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bitInfo)
            all_bit_info.append(bitInfo)
            for x in fingerprint.GetOnBits():
                for i in range(len(bitInfo[x])):
                    list_bits.append((mol,x,bitInfo,i))
                    legends.append(str(x))

        elif method == 'RDKit':
            bitInfo = {}
            fingerprint = Chem.RDKFingerprint(mol, maxPath=radius, fpSize=n_bits, bitInfo = bitInfo)
            bit_tuples = [(mol, k, bitInfo) for k in bitInfo.keys()]
            all_bit_info.extend(bit_tuples)
            for x in fingerprint.GetOnBits():
                legends.append(str(x))

        elif method == 'MACCS':
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
        elif method == 'TopologicalTorsion':
            fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        elif method == 'AtomPair':
            fingerprint = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        else:
            raise ValueError(f"Unsupported fingerprint method: {method}")

        fingerprints.append(np.array(fingerprint))

    if return_bit_info:
        return np.array(fingerprints), all_bit_info, list_bits, legends
    return np.array(fingerprints)


def train_validate(target_property: str = 'Tg', test_size: float = 0.2, n_jobs: int = -1) -> tuple:
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


    # Create results directory if it doesn't exist
    results_dir = f'./results/{target_property}/'
    os.makedirs(results_dir, exist_ok=True)
    # Setup logging
    setup_logging(results_dir)
    set_global_random_seed(42)

    if target_property == 'Tg':
        data_path = './data/labeled/polyinfo/Tg_SMILES_class_pid_polyinfo_median.csv'
    data = pd.read_csv(data_path)
    smiles = data['SMILES']
    y = data[target_property].to_numpy()
    # Check for NaN or infinite values in y
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Target variable contains NaN or infinite values.")

    X = smiles_to_fingerprint(smiles)
    # Check for NaN or infinite values in X
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Feature matrix contains NaN or infinite values.")

    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, 
                                                                                   test_size=test_size, random_state=42)
    
    # Verify shapes
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Save the split datasets
    np.save(os.path.join(results_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(results_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(results_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(results_dir, 'y_test.npy'), y_test)
    smiles_train.to_csv(os.path.join(results_dir, 'smiles_train.csv'), index=False)
    smiles_test.to_csv(os.path.join(results_dir, 'smiles_test.csv'), index=False)

    automl = TPOTRegressor(generations=3, population_size=10, cv=5,
                           verbosity=2, random_state=42, n_jobs=n_jobs)
    automl.fit(X_train, y_train)

    y_pred_train = automl.predict(X_train)
    y_pred_test = automl.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f'Train RMSE: {train_rmse:.4f}')
    print(f'Train R2: {train_r2:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Test R2: {test_r2:.4f}')

    # Parity plot with Nature journal style
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['figure.figsize'] = [8, 8]

    plt.figure()
    plt.scatter(y_train, y_pred_train, color='blue', label='Train Data', alpha=0.6, edgecolor='w', s=60)
    plt.scatter(y_test, y_pred_test, color='red', label='Test Data', alpha=0.6, edgecolor='w', s=60)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Parity Plot for {target_property} Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parity_plot.png'))
    plt.close()

    # Save the best model and the pipeline
    automl.export(os.path.join(results_dir, 'best_model.py'))
    joblib.dump(automl.fitted_pipeline_, os.path.join(results_dir, 'best_model.pkl'))

    return automl, X_train, X_test, y_train, y_test, y_pred_test

def train_validate_uq(fp_method: str ='Morgan', radius: int = 2, n_bits: int = 2048,
                      target_property: str = 'Tg', 
                      test_size: float = 0.2, n_jobs: int = -1,
                      model_type: str = 'QuantileRandomForest') -> tuple:
    results_dir = f'./results/{target_property}_uq/{model_type}_{fp_method}_{radius}_{n_bits}/'
    os.makedirs(results_dir, exist_ok=True)
    setup_logging(results_dir)

    if target_property == 'Tg':
        data_path = './data/labeled/polyinfo/Tg_SMILES_class_pid_polyinfo_median.csv'
    data = pd.read_csv(data_path)
    smiles = data['SMILES']
    y = data[target_property].to_numpy()
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Target variable contains NaN or infinite values.")

    # X = smiles_to_fingerprint(smiles)
    X = smiles_to_fingerprint(smiles, fp_method, radius, n_bits)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Feature matrix contains NaN or infinite values.")

    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=test_size, random_state=42)

    # Save the split datasets
    np.save(os.path.join(results_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(results_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(results_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(results_dir, 'y_test.npy'), y_test)
    smiles_train.to_csv(os.path.join(results_dir, 'smiles_train.csv'), index=False)
    smiles_test.to_csv(os.path.join(results_dir, 'smiles_test.csv'), index=False)

    if model_type == 'QuantileRandomForest':
        # Using RandomForestQuantileRegressor for uncertainty quantification
        model = RandomForestQuantileRegressor(random_state=42, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        # Uncertainty prediction
        lower_quantile = model.predict(X_test, quantiles=0.05)
        upper_quantile = model.predict(X_test, quantiles=0.95)
        std_dev = (upper_quantile - lower_quantile) / 2  # Approximate std as half the prediction interval

        lower_quantile_tr = model.predict(X_train, quantiles=0.05)
        upper_quantile_tr = model.predict(X_train, quantiles=0.95)
        std_dev_tr = (upper_quantile_tr - lower_quantile_tr) / 2
    # TODO: add more  U-model
    elif model_type == 'DropoutMLP':
        model = train_mlp_model(X_train, y_train)
        y_pred_test, _, quantile_predictions = mlp_predict_with_uncertainty(
            model, X_test, n_iter=100, quantiles=[5, 95])
        lower_quantile = quantile_predictions[0]
        upper_quantile = quantile_predictions[1]
        std_dev = (upper_quantile - lower_quantile) / 2

        y_pred_train, _, quantile_predictions_tr = mlp_predict_with_uncertainty(
            model, X_train, n_iter=100, quantiles=[5, 95])
        lower_quantile_tr = quantile_predictions_tr[0]
        upper_quantile_tr = quantile_predictions_tr[1]
        std_dev_tr = (upper_quantile_tr - lower_quantile_tr) / 2
    elif model_type == 'BNN':
        X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        # y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)
        
        input_shape = (X_train_tf.shape[1],)
        hidden_units = 64

        model = build_bayesian_model(input_shape=input_shape, hidden_units=hidden_units, output_shape=1)
        model = compile_and_train_BNN(model, X_train_tf, y_train_tf, num_epochs=100)
        # Make predictions with uncertainty on the test data
        y_pred_test, _, quantile_predictions = bnn_predict_with_uncertainty(
            model, X_test_tf, n_iter=100, quantiles=[5, 95])
        lower_quantile = quantile_predictions[0]
        upper_quantile = quantile_predictions[1]
        std_dev = (upper_quantile - lower_quantile) / 2

        y_pred_train, _, quantile_predictions_tr = bnn_predict_with_uncertainty(
            model, X_train_tf, n_iter=100, quantiles=[5, 95])
        lower_quantile_tr = quantile_predictions_tr[0]
        upper_quantile_tr = quantile_predictions_tr[1]
        std_dev_tr = (upper_quantile_tr - lower_quantile_tr) / 2

    else:
        raise ValueError("Unsupported model type!")
    

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f'Train RMSE: {train_rmse:.4f}')
    print(f'Train R2: {train_r2:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Test R2: {test_r2:.4f}')

    # Parity plot with uncertainty
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['figure.figsize'] = [8, 8]

    plt.figure()
    # plt.scatter(y_test, y_pred_test, color='red', label='Test Data', alpha=0.6, edgecolor='w', s=60)
    # plt.fill_between(np.arange(len(y_test)), lower_quantile, upper_quantile, color='gray', alpha=0.2, label='Prediction Interval')
    
    # Plot Train Data
    # plt.scatter(y_train, y_pred_train, color='blue', label='Train Data', alpha=0.6, edgecolor='w', s=60)
    plt.errorbar(y_train, y_pred_train, yerr=[y_pred_train - lower_quantile_tr, 
                                            upper_quantile_tr - y_pred_train],
                fmt='o', color='blue', alpha=0.4, label='Train')
    # Plot Test Data
    # plt.scatter(y_test, y_pred_test, color='red', label='Test Data', alpha=0.6, edgecolor='w', s=60)
    # plt.errorbar(y_test, y_pred_test, yerr=[y_pred_test - lower_quantile, upper_quantile - y_pred_test], 
    #             fmt='o', color='red', alpha=0.4, label='Test')
    
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Parity Plot with Uncertainty for {target_property}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parity_plot_with_uncertainty_tr.png'))
    plt.close()

    plt.figure()
    # plt.scatter(y_test, y_pred_test, color='red', label='Test Data', alpha=0.6, edgecolor='w', s=60)
    # plt.fill_between(np.arange(len(y_test)), lower_quantile, upper_quantile, color='gray', alpha=0.2, label='Prediction Interval')
    
    # Plot Train Data
    # plt.scatter(y_train, y_pred_train, color='blue', label='Train Data', alpha=0.6, edgecolor='w', s=60)
    # plt.errorbar(y_train, y_pred_train, yerr=[y_pred_train - qrf.predict(X_train, quantiles=0.05), 
    #                                         qrf.predict(X_train, quantiles=0.95) - y_pred_train],
    #             fmt='o', color='blue', alpha=0.2, label='Train')
    # Plot Test Data
    # plt.scatter(y_test, y_pred_test, color='red', label='Test Data', alpha=0.6, edgecolor='w', s=60)
    plt.errorbar(y_test, y_pred_test, yerr=[y_pred_test - lower_quantile, upper_quantile - y_pred_test], 
                fmt='o', color='red', alpha=0.4, label='Test')
    
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Parity Plot with Uncertainty for {target_property}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parity_plot_with_uncertainty_te.png'))
    plt.close()

    # test data
    # Evaluate uncertainty: Sparsification plot and Spearman’s rank correlation coefficient
    abs_errors = np.abs(y_test - y_pred_test)
    spearman_corr, _ = spearmanr(abs_errors, std_dev)
    print(f"Spearman's rank correlation coefficient (error vs. std dev) - Test: {spearman_corr:.4f}")

    # Sparsification plot
    sorted_indices = np.argsort(std_dev) # from small to large
    sorted_errors = abs_errors[sorted_indices]
    sorted_std_dev = std_dev[sorted_indices]
    cumulative_errors = np.cumsum(sorted_errors)
    cumulative_std_dev = np.cumsum(sorted_std_dev)

    plt.figure()
    plt.plot(np.arange(len(cumulative_errors)), cumulative_errors, label='Cumulative Error')
    plt.plot(np.arange(len(cumulative_std_dev)), cumulative_std_dev, label='Cumulative Std Dev', linestyle='--')
    plt.xlabel('Sorted Samples')
    plt.ylabel('Cumulative Value')
    plt.title('Sparsification Plot (test)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sparsification_plot_test.png'))
    plt.close()

    # train data
    # Evaluate uncertainty: Sparsification plot and Spearman’s rank correlation coefficient
    abs_errors_tr = np.abs(y_train - y_pred_train)
    spearman_corr_tr, _ = spearmanr(abs_errors_tr, std_dev_tr)
    print(f"Spearman's rank correlation coefficient (error vs. std dev) - Train: {spearman_corr_tr:.4f}")

    # Sparsification plot
    sorted_indices_tr = np.argsort(std_dev_tr) # from small to large
    sorted_errors_tr = abs_errors_tr[sorted_indices_tr]
    sorted_std_dev_tr = std_dev_tr[sorted_indices_tr]
    cumulative_errors_tr = np.cumsum(sorted_errors_tr)
    cumulative_std_dev_tr = np.cumsum(sorted_std_dev_tr)

    plt.figure()
    plt.plot(np.arange(len(cumulative_errors_tr)), cumulative_errors_tr, label='Cumulative Error')
    plt.plot(np.arange(len(cumulative_std_dev_tr)), cumulative_std_dev_tr, label='Cumulative Std Dev', linestyle='--')
    plt.xlabel('Sorted Samples')
    plt.ylabel('Cumulative Value')
    plt.title('Sparsification Plot (Train)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sparsification_plot_train.png'))
    plt.close()

    # Save the model
    joblib.dump(model, os.path.join(results_dir, 'best_model_with_uq.pkl'))

    return model, X_train, X_test, y_train, y_test, y_pred_test, lower_quantile, upper_quantile


if __name__ == "__main__":
    args = parse_args()
    # Suppress UserWarning messages
    warnings.filterwarnings("ignore", category=UserWarning)
    # train_validate(args.target_property, args.test_size, args.n_jobs)
    train_validate_uq(fp_method=args.fpmethod, radius = args.radius, n_bits=args.n_bits,
                      target_property = args.target_property, 
                      test_size = args.test_size, n_jobs = args.n_jobs,
                      model_type = args.model)
