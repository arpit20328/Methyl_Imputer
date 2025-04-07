import argparse
from utils import load_data, sparsify_data, find_optimal_k, knn_impute, save_outputs

def main():
    parser = argparse.ArgumentParser(description="Sparse + Impute Tool for Methylation/Omics Matrix")
    parser.add_argument("input_file", type=str, help="Path to input TSV file")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold (%) to remove sparse columns")
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 15, 20], help="List of K values for KNN imputation")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")

    args = parser.parse_args()

    print("🔄 Loading Data...")
    df = load_data(args.input_file)

    print("💡 Sparsifying Data...")
    df_sparse, removed_cols, sparsity_info = sparsify_data(df, args.threshold)

    print(f"🧠 Performing KNN Imputation over K = {args.k_values}")
    optimal_k, rmse_table, imputed_matrix = find_optimal_k(df_sparse, args.k_values, args.threads)

    print("✅ Imputation Done. Saving Outputs...")
    save_outputs(df_sparse, imputed_matrix, rmse_table)

    print(f"\n🎉 Process Completed! Optimal K = {optimal_k}\n")

if __name__ == "__main__":
    main()
