from datasets import load_dataset
import os
import json
from tqdm import tqdm
import re

save_dir = "./data/"
os.makedirs(save_dir, exist_ok=True)

def load_wikipedia_dataset(
        split: str = "train",
        data_dir: str = "20231101.simple",
    ):
    dataset = load_dataset("wikimedia/wikipedia", data_dir, split=split)

    save_folder = os.path.join(save_dir, "wikimedia--wikipedia", data_dir, "seed")
    os.makedirs(save_folder, exist_ok=True)
    text_lens = []
    sentence_lens = []
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {data_dir} to {save_folder}"):
        text: str = row.get("text", "")
        text_lens.append(len(text.split()))
        sentence_lens.append(len(re.findall(r".+?[.!?]+", text)))
        file_name = f"row_{i}.txt"
        with open(os.path.join(save_folder, file_name), "w", encoding="utf-8") as f:
            f.write(text)
    print(f"Average Text Length: {sum(text_lens)/len(text_lens)}")
    print(f"Min Text Length: {min(text_lens)}")
    print(f"Max Text Length: {max(text_lens)}")
    print(f"Average Sentences: {sum(sentence_lens)/len(sentence_lens)}")
    print(f"Min Sentences: {min(sentence_lens)}")
    print(f"Max Sentences: {max(sentence_lens)}")

if __name__ == "__main__":
    load_wikipedia_dataset(split="train")