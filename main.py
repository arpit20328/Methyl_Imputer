import argparse
from utils import load_data, sparsify_data, find_optimal_k, save_outputs
import pandas as pd


def print_threshold_analysis(df):
    na_counts = df.isna().sum()
    na_matrix = pd.DataFrame({"Column_Name": na_counts.index, "NA_Counts": na_counts.values})

    K_values = list(range(1, 31))
    num_rows = df.shape[0]
    num_cols = df.shape[1]

    print("\n📊 Threshold Impact Analysis")
    print("Total Rows:", num_rows)
    print("Total Columns:", num_cols)

    N_values = [round((K / 100) * num_rows, 2) for K in K_values]

    print("\nK% Threshold | N (Rows) | Columns Removed | Data Loss %")
    print("------------------------------------------------------")
    for K, N in zip(K_values, N_values):
        filtered_na_matrix = na_matrix[na_matrix["NA_Counts"] > N]
        num_columns_above_K = filtered_na_matrix.shape[0]
        data_loss_percentage = (num_columns_above_K / num_cols) * 100

        print(f"    {K:>2}%       | {N:>7} | {num_columns_above_K:>15} | {data_loss_percentage:>10.2f}%")


def main():
    parser = argparse.ArgumentParser(description="🔬 Sparse + Impute Tool for Omics/Methylation Matrix")
    parser.add_argument("input_file", type=str, help="Path to input TSV file")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold percentage to drop sparse columns (leave blank to choose interactively)")
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 15, 20], help="List of K values for KNN imputation (default: 5 10 15 20)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing (default: 4)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files (default: output)")

    args = parser.parse_args()

    print("🔄 Step 1: Loading Data...")
    df = load_data(args.input_file)

    if args.threshold is None:
        print_threshold_analysis(df)
        while True:
            try:
                user_threshold = float(input("\n📥 Enter your desired threshold percentage for sparsification (e.g. 1.0): "))
                if 0 <= user_threshold <= 100:
                    break
                else:
                    print("❌ Please enter a value between 0 and 100.")
            except ValueError:
                print("❌ Invalid input. Please enter a numeric value.")
        args.threshold = user_threshold

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
