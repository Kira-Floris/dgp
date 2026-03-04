# python -m scripts.parallel_synthetic_data_to_csv ./data/wikimedia--wikipedia--gemma/20231101.simple -t 0.8 -s comet_score --src_lang english --tgt_lang kinyarwanda -o results/wikimedia--wikipedia--gemma-20231101-simple_results.csv
# python -m scripts.parallel_synthetic_data_to_csv ./data/wikimedia--wikipedia/20231101.simple -t 0.8 -s comet_score --src_lang english --tgt_lang kinyarwanda -o results/wikimedia--wikipedia-20231101-simple_results.csv
# python -m scripts.parallel_synthetic_data_to_csv ./data/mbazanlp--kinyarwanda_monolingual_v01.1 -t 0.8 -s comet_score --src_lang kinyarwanda --tgt_lang english -o results/mbazanlp--kinyarwanda_monolingual_v01.1_results.csv
import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def create_parallel_translations(main_path: str, threshold: int, score_column: str="comet_score")->pd.DataFrame:
    translation_path = os.path.join(main_path, "translation")
    original_path = os.path.join(main_path, "seed")
    tsv_path = os.path.join(main_path, "tsv_results")
    
    tsv_files = os.listdir(tsv_path)
    dfs = []
    for tsv_file in tqdm(tsv_files, desc=f"Collecting Translations above {threshold} for {score_column} Column", total=len(tsv_files)):
        df = pd.read_csv(os.path.join(tsv_path, tsv_file), sep="\t")
        avg = df[score_column].mean()
        above_threshold_df = df[df[score_column]>=threshold]
        if avg > threshold:
            if len(above_threshold_df)>0:
                dfs.append(above_threshold_df)
    
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

def df_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Combine Translations Higher than a Certain Threshold"
    )

    parser.add_argument(
        "folder",
        help='Path to folder containing TSV files'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.8,
        help='Threshold to select sentences for.'
    )
    parser.add_argument(
        '-s', '--score_column',
        type=str,
        default="comet_score",
        help="TSV column name for the threshold."
    )
    parser.add_argument(
        '--src_lang',
        type=str,
        help="Language for the seed folder",
    )
    parser.add_argument(
        '--tgt_lang',
        type=str,
        help="Language for the translation folder",
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file for detailed results',
        default=None
    )

    args = parser.parse_args()

    try:
        df = create_parallel_translations(
            args.folder,
            args.threshold,
            args.score_column
        )
        df["src_lang"] = args.src_lang
        df["tgt_lang"] = args.tgt_lang

        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"DataFrame saved at {args.output}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise e

if __name__=="__main__":
    import sys

    if len(sys.argv) == 1:
        print("Usage Examples:")
        print("  python -m scripts.parallel_synthetic_data_to_csv ./data/wikimedia--wikipedia--gemma/20231101.simple -t 0.8 -s comet_score --src_lang english --tgt_lang kinyarwanda -o results/wikimedia--wikipedia--gemma-20231101-simple_results.csv")
        sys.exit(1)
    
    sys.exit(main())


# # arguments
# main_path = "./data/wikimedia--wikipedia--gemma/20231101.simple"
# threshold = 0.8
# score_column = "comet_score"
# source_lang = "english"
# target_lang = "kinyarwanda"
# save_path = "results/wikimedia--wikipedia--gemma-20231101-simple_results.csv"


# df = create_parallel_translations(main_path, threshold, score_column)

# df["source_lang"] = source_lang
# df["target_lang"] = target_lang

# df.to_csv(save_path, index=False, encoding="utf-8")
