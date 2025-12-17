import os
import shutil
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm

def filter_and_move_seed_files(
    base_dir: str,
    threshold: float,
    results_dir: str = "tsv_results",
    seed_dir: str = "seed",
    output_dir: str = "pretraining",
) -> List[str]:
    """
    1. Loop through tsv_results
    2. Identify files with any comet_score >= threshold
    3. Exclude those files from seed_translation
    4. Move remaining seed files to pretraining/
    
    Returns:
        List of filenames that were ABOVE threshold
    """

    base_dir = Path(base_dir)
    results_path = base_dir / results_dir
    seed_path = base_dir / seed_dir
    output_path = base_dir / output_dir

    output_path.mkdir(exist_ok=True)

    tsv_above_threshold = []

    # --- Step 1: scan tsv_results ---
    for tsv_file in tqdm(results_path.glob("*.tsv"), desc="Processing TSV files", total=len(list(results_path.glob("*.tsv")))):
        try:
            df = pd.read_csv(tsv_file, sep="\t")

            if "comet_score" not in df.columns:
                print(f"[WARN] comet_score not found in {tsv_file.name}")
                continue
            above_threshold = df[df["comet_score"] >= threshold]
            # print(len(above_threshold))
            if len(above_threshold)>0:
                tsv_above_threshold.append(tsv_file.name)

        except Exception as e:
            print(f"[ERROR] Failed to process {tsv_file.name}: {e}")
    # print(tsv_above_threshold)

    # --- Step 2: move remaining seed files ---
    for seed_file in tqdm(seed_path.glob("*.txt"), desc="Moving seed files", total=len(list(seed_path.glob("*.txt")))):
        if seed_file.name not in tsv_above_threshold:
            destination = output_path / seed_file.name
            if os.path.exists(destination):
                continue
            shutil.copy(str(seed_file), destination)

    print(f"Files ABOVE threshold ({threshold}): {len(tsv_above_threshold)}")
    print(f"Files in seed directory: {len(list(seed_path.glob('*.txt')))}")
    print(f"Files moved to {output_dir}: {len(list(output_path.glob('*.tsv')))}")

    return tsv_above_threshold

base_dir = "./data/mbazanlp--kinyarwanda_monolingual_v01.1"
threshold = 0.6
if __name__ == "__main__":
    filter_and_move_seed_files(base_dir=base_dir, threshold=threshold)