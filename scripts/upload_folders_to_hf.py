import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm

def upload_txt_dir_to_hf(
    txt_dir: str,
    repo_id: str,
    split_test_size: float = 0.001,
    private: bool = False,
):
    """
    Upload a directory of .txt files as a Hugging Face dataset (train split).
    """

    txt_dir = Path(txt_dir)
    assert txt_dir.exists(), f"{txt_dir} does not exist"

    texts = []
    file_id = []

    for txt_file in tqdm(sorted(txt_dir.glob("*.txt")), desc="Reading .txt files", total=len(list(txt_dir.glob("*.txt")))):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        texts.append(content)
        file_id.append(int(txt_file.name.split(".txt")[0].split("row_")[-1]))
    
    # print(file_id[:5])

    dataset = Dataset.from_dict({
        "text": texts,
        "id": file_id
    })

    splits = dataset.train_test_split(test_size=split_test_size, seed=42)

    dataset_dict = DatasetDict({
        "train": splits["train"],
        "dev": splits["test"]
    })

    dataset_dict.push_to_hub(
        repo_id,
        private=private
    )

    print(f"✅ Uploaded {len(dataset)} examples to {repo_id} (train split)")


if __name__ == "__main__":
    upload_txt_dir_to_hf(
        txt_dir="./data/mbazanlp--kinyarwanda_monolingual_v01.1/pretraining",
        repo_id="Kira-Floris/kinyarwanda_monolingual_v01.1",
        split_test_size=0.001,
        private=True
    )
