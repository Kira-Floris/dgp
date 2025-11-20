from datasets import load_dataset
import os
import json

save_dir = "./data/"
os.makedirs(save_dir, exist_ok=True)

def load_wikipedia_dataset(split: str = "train", data_dir: str = "20231101.en"):
    """
    Load the Wikipedia dataset from Hugging Face Datasets.
    
    Args:
        split: Dataset split to load (e.g., "train", "validation", "test")
        
    Returns:
        Loaded dataset split
    """
    dataset = load_dataset("wikimedia/wikipedia", data_dir, split=split)
    # save the dataset to ./data/ only saving the text column as json file
    jsonl_path = os.path.join(save_dir, f"wikipedia_{split}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in dataset:
            text = row.get("text", "")
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Saved text column to {jsonl_path}")

    return dataset

if __name__ == "__main__":
    load_wikipedia_dataset(split="train")