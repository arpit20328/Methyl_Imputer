import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import multiprocessing
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        first_column = df.iloc[:, 0]
        df = df.iloc[:, 1:]
        df.index = first_column
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading file: {e}")

def calculate_sparsity(df):
    return 100 * df.isna().sum().sum() / (df.shape[0] * df.shape[1])

def sparsify_data(df, threshold_percent):
    threshold = (threshold_percent / 100) * df.shape[0]
    na_counts = df.isna().sum()
    columns_to_remove = na_counts[na_counts > threshold].index.tolist()
    df_sparse = df.drop(columns=columns_to_remove)

    original_sparsity = calculate_sparsity(df)
    new_sparsity = calculate_sparsity(df_sparse)
    sparsity_info = {
        "original_sparsity": original_sparsity,
        "new_sparsity": new_sparsity,
        "reduction": original_sparsity - new_sparsity
    }
    return df_sparse, columns_to_remove, sparsity_info

def evaluate_k(k, beta_values, non_missing_mask):
    imputer = KNNImputer(n_neighbors=k)
    imputed_matrix = imputer.fit_transform(beta_values)
    rmse = np.sqrt(mean_squared_error(
        beta_values[non_missing_mask].ravel(),
        imputed_matrix[non_missing_mask].ravel()
    ))
    return k, rmse

def find_optimal_k(df, k_list, threads):
    beta_values = df.to_numpy()
    non_missing_mask = ~np.isnan(beta_values)

    with multiprocessing.Pool(processes=threads) as pool:
        rmse_results = pool.starmap(evaluate_k, [(k, beta_values, non_missing_mask) for k in k_list])

    rmse_values = [rmse for _, rmse in rmse_results]
    optimal_k = rmse_results[np.argmin(rmse_values)][0]

    rmse_table = pd.DataFrame(rmse_results, columns=["K Value", "RMSE"])
    return optimal_k, rmse_table, knn_impute(df, optimal_k)

def knn_impute(df, k):
    imputer = KNNImputer(n_neighbors=k)
    imputed_values = imputer.fit_transform(df)
    return pd.DataFrame(imputed_values, index=df.index, columns=df.columns)

def save_outputs(df_sparse, imputed_matrix, rmse_table, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    df_sparse.to_csv(os.path.join(output_dir, "sparse_matrix.tsv"), sep="\t")
    imputed_matrix.to_csv(os.path.join(output_dir, "imputed_matrix.tsv"), sep="\t")
    rmse_table.to_csv(os.path.join(output_dir, "rmse_results.tsv"), sep="\t", index=False)
    print(f"Outputs saved to '{output_dir}' folder.")
