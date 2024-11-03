import os
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import argparse
from utils import set_global_random_seed, extract_maccs_feature_descriptions_list, remove_and_sort_duplicates
from train import smiles_to_fingerprint
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training and validating models.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Explain models for polymer property prediction.')
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data.')
    parser.add_argument('--fpmethod', type=str, default='Morgan', choices=['Morgan', 'RDKit', 'MACCS', 'TopologicalTorsion', 'AtomPair'], help='Fingerprinting method.')
    parser.add_argument('--radius', type=int, default=2, help='Radius for Morgan and Topological Torsion.')
    parser.add_argument('--n_bits', type=int, default=2048, help='Number of bits for the fingerprint vector.')

    parser.add_argument('--target_property', type=str, default='Tg', help='Target property to predict (e.g., Tg).')

    return parser.parse_args()

def get_final_model(model):
    """
    Get the final estimator from the pipeline or the model itself if it's not a pipeline.
    
    Args:
        model: The machine learning model or pipeline.

    Returns:
        The final model (e.g., the classifier/regressor itself).
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model

def preprocess_data(pipeline, X):
    """
    Apply the preprocessing steps of the pipeline to the data.
    
    Args:
        pipeline: The pipeline containing preprocessing steps.
        X: The input data to preprocess.

    Returns:
        The preprocessed data.
    """
    if isinstance(pipeline, Pipeline):
        X_transformed = X
        # Apply all steps except the final model step
        for name, step in pipeline.steps[:-1]:
            X_transformed = step.transform(X_transformed)
        return X_transformed
    return X

def is_tree_based(model):
    """
    Check if the model is tree-based (e.g., decision tree, random forest, gradient boosting).
    
    Args:
        model: The machine learning model or pipeline.

    Returns:
        bool: True if the model is tree-based, False otherwise.
    """
    tree_based_models = (
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "CatBoostClassifier", "CatBoostRegressor"
    )
    
    final_model = get_final_model(model)
    
    return final_model.__class__.__name__ in tree_based_models

def explain_model(results_dir, target_property, fp_method='Morgan', radius=2, n_bits=2048):
    """
    Generate SHAP values to explain the model's predictions.
    
    Args:
        results_dir: Directory where the model and data are saved.
        target_property: The name of the target property being predicted.
    """
    set_global_random_seed(42)
    # Load the saved data
    X_train = np.load(os.path.join(results_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(results_dir, 'X_test.npy'))
    # y_train = np.load(os.path.join(results_dir, 'y_train.npy'))
    # y_test = np.load(os.path.join(results_dir, 'y_test.npy'))

    # Load the trained pipeline
    # pipeline = joblib.load(os.path.join(results_dir, 'best_model.pkl'))
    pipeline = joblib.load(os.path.join(results_dir, 'best_model_with_uq.pkl'))

    X_train_transformed = preprocess_data(pipeline, X_train)
    X_test_transformed = preprocess_data(pipeline, X_test)
    final_model = get_final_model(pipeline)

    # Determine the appropriate SHAP explainer
    if is_tree_based(final_model):
        explainer = shap.TreeExplainer(final_model)
        print(f"Using TreeExplainer for model: {final_model.__class__.__name__}")
    else:
        # TODO: if-else on polymer representation
        explainer = shap.KernelExplainer(final_model.predict, shap.kmeans(X_train_transformed, 10), 
                                         feature_names = extract_maccs_feature_descriptions_list())
        # print(np.shape(X_train_transformed), np.shape(extract_maccs_feature_descriptions_list()))
        print(f"Using KernelExplainer for model: {final_model.__class__.__name__}")
    
    # Additional visualization details for Morgan if necessary
    if fp_method == 'Morgan':
        smiles_df = pd.read_csv(os.path.join(results_dir, 'smiles_test.csv'))
        smiles_test = smiles_df.iloc[:, 0].tolist()
        test_morgan_fp, test_bit_info, list_bits, legends = smiles_to_fingerprint(smiles_test, method='Morgan', 
                                                                                  radius=radius, n_bits=n_bits, return_bit_info=True)
        legends, list_bits = remove_and_sort_duplicates(legends, list_bits)
        # Drawing the Morgan bits (cannot hold in one img, too large)
        # Process in batches of 100
        batch_size = 100
        for i in range(0, len(legends), batch_size):
            # Get the batch of each list
            batch_list_bits = list_bits[i:i+batch_size]
            batch_legends = legends[i:i+batch_size]
            img = Draw.DrawMorganBits(batch_list_bits, molsPerRow=4, legends=batch_legends, subImgSize=(150, 150))
            # Save the image to a file
            img.save(os.path.join(results_dir, 'morgan_bits_summary_{}.png'.format(i//batch_size + 1)))
    
    # Additional visualization details for RDKitFP if necessary
    elif fp_method == 'RDKit':
        smiles_df = pd.read_csv(os.path.join(results_dir, 'smiles_test.csv'))
        smiles_test = smiles_df.iloc[:, 0].tolist()
        test_morgan_fp, test_bit_info, list_bits, legends = smiles_to_fingerprint(smiles_test, method='RDKit', 
                                                                                  radius=radius, n_bits=n_bits, return_bit_info=True)
        legends, test_bit_info = remove_and_sort_duplicates(legends, test_bit_info)
        # Drawing the RDKit bits (cannot hold in one img, too large)
        # Process in batches of 100
        batch_size = 100
        for i in range(0, len(legends), batch_size):
            # Get the batch of each list
            batch_list_bits = test_bit_info[i:i+batch_size]
            batch_legends = legends[i:i+batch_size]
            img = Draw.DrawRDKitBits(batch_list_bits, molsPerRow=4, legends=batch_legends, subImgSize=(150, 150))
            # Save the image to a file
            img.save(os.path.join(results_dir, 'RDKit_bits_summary_{}.png'.format(i//batch_size + 1)))

    # X_test_transformed = X_test_transformed[:10]
    # Generate SHAP values
    shap_values = explainer.shap_values(X_test_transformed)
    
    # Global Explanation
    if fp_method == "MACCS":
        shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", 
                        feature_names=extract_maccs_feature_descriptions_list(), show=False)
        plt.savefig(os.path.join(results_dir, 'shap_summary_plot_bar.png'))
        plt.close()
        shap.summary_plot(shap_values, X_test_transformed, 
                        feature_names=extract_maccs_feature_descriptions_list(), show=False)
        plt.savefig(os.path.join(results_dir, 'shap_summary_plot_beeswarm.png'))
        plt.close()
    if fp_method in ("Morgan", "RDKit"):
        feature_names = [f'Bit {i}' for i in range(len(X_test_transformed[0]))]
        shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", 
                        feature_names=feature_names, show=False)
        plt.savefig(os.path.join(results_dir, 'shap_summary_plot_bar.png'))
        plt.close()
        shap.summary_plot(shap_values, X_test_transformed, 
                        feature_names=feature_names, show=False)
        plt.savefig(os.path.join(results_dir, 'shap_summary_plot_beeswarm.png'))
        plt.close()

    # Local Explanation for the first few samples
    # for i in range(3):  # Show explanation for the first 3 test cases
    #     shap.force_plot(explainer.expected_value, shap_values[i], X_test_transformed[i], matplotlib=True, show=False)
    #     plt.savefig(os.path.join(results_dir, f'shap_force_plot_{i}.png'))
    #     plt.close()
    for i in range(3):  
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        if fp_method == "MACCS":
            shap.waterfall_plot(shap.Explanation(values=shap_values[i], 
                                                base_values=explainer.expected_value, 
                                                data=X_test_transformed[i],
                                                feature_names=extract_maccs_feature_descriptions_list()),
                                                show=False)
        elif fp_method in ("Morgan", "RDKit"):
            feature_names = [f'Bit {i}' for i in range(len(X_test_transformed[0]))]
            shap.waterfall_plot(shap.Explanation(values=shap_values[i], 
                                                base_values=explainer.expected_value, 
                                                data=X_test_transformed[i],
                                                feature_names=feature_names),
                                                show=False)
        plt.tight_layout()  # Ensure the layout fits well within the figure
        plt.savefig(os.path.join(results_dir, f'shap_waterfall_plot_{i}.png'))
        plt.close()
    
    # # also saved the chem structure viz of the top bits of each local data
    # for j in range(3):
    #     # Additional visualization for Morgan fingerprints using Draw.DrawMorganBit()
    #     if fp_method == 'Morgan':
    #         top_bits = np.argsort(-np.abs(shap_values[j]))[:10]  # Top 10 influential bits for this instance
    #         mol = Chem.MolFromSmiles(smiles_test[j])
    #         bit_info = {}
    #         fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
    #         # fp = test_morgan_fp[j]
    #         # bit_info = test_bit_info[j]
    #         # Prepare to visualize only "on" bits among the top SHAP important bits
    #         list_bits = []
    #         legends = []
    #         for bit in top_bits:
    #             if bit in bit_info and bit in fp.GetOnBits():  # Check if the bit is "on"
    #                 atom_indices = [x[0] for x in bit_info[bit]]
    #                 highlight = list(set(atom_indices))  # Get unique indices to highlight
    #                 list_bits.append((mol, bit, bit_info, highlight))  # Prepare tuple for visualization
    #                 legends.append(f"Bit {bit}")  # Legend for each bit
    #             # img = Draw.DrawMorganBit(mol, top_bit, test_bit_info[j], useSVG=True)
    #             # # Save or display the image
    #             # with open(os.path.join(results_dir, f'mol_bit_{top_bit}_test_{j}.svg'), 'w') as f_svg:
    #             #     f_svg.write(img)
    #             # Visualize the bits using Draw.DrawMorganBits
    #         if list_bits:  # Ensure there are bits to draw
    #             img = Draw.DrawMorganBits(list_bits, molsPerRow=4, legends=legends, subImgSize=(200, 200))
    #             img.save(os.path.join(results_dir, f'morgan_important_on_bits_test_{j}.png'))  # Save the visualization
    #         else:
    #             print("No 'on' bits among the top SHAP important bits for this instance.")

    print(f"SHAP explanations saved in {results_dir}")

if __name__ == "__main__":
    args = parse_args()
    # target_property = 'Tg'  # Example target property
    # results_dir = f'./results/{target_property}_uq/'
    results_dir = f'./results/{args.target_property}_uq/{args.fpmethod}_{args.radius}_{args.n_bits}/'
    # results_dir = f'./results/{target_property}/'
    
    explain_model(results_dir, args.target_property, args.fpmethod, args.radius, args.n_bits)
