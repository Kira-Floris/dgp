import pandas as pd

def combine_directions(file1, file2, output_file):
    # Read the two CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1["direction"] = file1.split("/")[-1].split(".")[0].replace("-","->")
    df2["direction"] = file2.split("/")[-1].split(".")[0].replace("-","->")

    # Combine the DataFrames by concatenating them
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    combine_directions(
        "results/fleurs_mt_benchmark_gemma3_27b/en-rw.csv",
        "results/fleurs_mt_benchmark_gemma3_27b/rw-en.csv",
        "results/fleurs_mt_benchmark_gemma3_27b/combined.csv"
    )