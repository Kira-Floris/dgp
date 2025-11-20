import sacrebleu
import string
from comet import load_from_checkpoint
from huggingface_hub import snapshot_download

def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing and removing punctuation.
    
    Args:
        text: Input sentence
    Returns:
        Normalized sentence
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def compute_chrf(reference: str, hypothesis: str) -> float:
    """
    Compute ChrF score between reference and hypothesis.
    
    Args:
        reference: Original sentence
        hypothesis: Back-translated sentence
        
    Returns:
        ChrF score (0-100)
    """
    chrf = sacrebleu.corpus_chrf([normalize_text(hypothesis)], [[normalize_text(reference)]])
    return chrf.score

# model = load_from_checkpoint("chrismazii/kinycomet_unbabel")
import shutil
import os

snapshot_download("chrismazii/kinycomet_unbabel", local_dir="./comet_model")
os.makedirs("./comet_model/checkpoints", exist_ok=True)
shutil.copy("./comet_model/KinyCOMET+Unbabel.ckpt", "./comet_model/checkpoints/KinyCOMET+Unbabel.ckpt")
shutil.copy("./comet_model/KinyCOMET+XLM-Roberta.ckpt", "./comet_model/checkpoints/KinyCOMET+XLM-Roberta.ckpt")
model = load_from_checkpoint("./comet_model/checkpoints/KinyCOMET+Unbabel.ckpt")

def compute_comet(source:str, reference: str, hypothesis: str) -> float:
    """
    Compute COMET score between reference and hypothesis.
    
    Args:
        reference: Original sentence
        hypothesis: Back-translated sentence
    Returns:
        COMET score (float) 
    """
    data = [{
        "src": normalize_text(source),
        "mt": normalize_text(hypothesis),
        "ref": normalize_text(reference)
    }]
    scores = model.predict(data, batch_size=1, gpus=0, progress_bar=False)
    return float(scores["system_score"])
