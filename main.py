import argparse
from utils import load_data, sparsify_data, find_optimal_k, save_outputs

def main():
    parser = argparse.ArgumentParser(description="🔬 Sparse + Impute Tool for Omics/Methylation Matrix")
    parser.add_argument("input_file", type=str, help="Path to input TSV file")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold percentage to drop sparse columns (default: 1.0%)")
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 15, 20], help="List of K values for KNN imputation (default: 5 10 15 20)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing (default: 4)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files (default: output)")

    args = parser.parse_args()

    print("🔄 Step 1: Loading Data...")
    df = load_data(args.input_file)

    print(f"\n💡 Step 2: Sparsifying Data with Threshold = {args.threshold}%")
    df_sparse, removed_cols, sparsity_info = sparsify_data(df, args.threshold)
    print(f"Removed {len(removed_cols)} columns. Sparsity reduced by {sparsity_info['reduction']:.2f}%")

    print("\n🧠 Step 3: Running KNN Imputation with RMSE Evaluation")
    optimal_k, rmse_table, imputed_matrix = find_optimal_k(df_sparse, args.k_values, args.threads)
    print(f"Optimal K found: {optimal_k}")

    print("\n💾 Step 4: Saving Results...")
    save_outputs(df_sparse, imputed_matrix, rmse_table, args.output_dir)

    print("\n✅ All Done! Check your output directory for results.")

if __name__ == "__main__":
    main()
