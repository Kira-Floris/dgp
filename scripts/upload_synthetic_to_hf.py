import os
import pandas as pd
import argparse
from datasets import Dataset

def df_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset

def upload_ds_to_hf(ds: Dataset, repo_id: str, repo_type:str="dataset", private: bool=True, exist_ok:bool=True):
    from huggingface_hub import HfApi, create_repo
    
    # create repo
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=exist_ok
    )

    # upload dataset to huggingface
    api = HfApi()
    ds.push_to_hub(repo_id)


# def rename_columns(df: pd.DataFrame):


files = [
    "results/wikimedia--wikipedia-20231101-simple_results.csv",
    "results/mbazanlp--kinyarwanda_monolingual_v01.1_results.csv"
]
dfs = []

for file in files:
    df = pd.read_csv(file)

    # rename columns
    source_lang = list(df["src_lang"].unique())[0]
    target_lang = list(df["tgt_lang"].unique())[0]

    df[source_lang] = df["original_sentence"]
    df[target_lang] = df["forward_translation"]
    df["score"] = df["comet_score"]
    
    df_to_upload = df[[source_lang, target_lang, "score", "temperature", "src_lang", "tgt_lang"]]
    dfs.append(df_to_upload)

# combine the dfs
df = pd.concat(dfs, ignore_index=True)

ds = df_to_hf_dataset(df)

upload_ds_to_hf(
    ds, 
    "Kira-Floris/NMT_Synthetic_Data", 
    repo_type="dataset", 
    private=True, 
    exist_ok=True
)