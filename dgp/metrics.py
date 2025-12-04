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
# import shutil
# import os

# snapshot_download("chrismazii/kinycomet_unbabel", local_dir="./comet_model")
# os.makedirs("./comet_model/checkpoints", exist_ok=True)
# shutil.copy("./comet_model/KinyCOMET+Unbabel.ckpt", "./comet_model/checkpoints/KinyCOMET+Unbabel.ckpt")
# shutil.copy("./comet_model/KinyCOMET+XLM-Roberta.ckpt", "./comet_model/checkpoints/KinyCOMET+XLM-Roberta.ckpt")
# model = load_from_checkpoint("./comet_model/checkpoints/KinyCOMET+Unbabel.ckpt")

# def compute_comet(source:str, reference: str, hypothesis: str) -> float:
#     """
#     Compute COMET score between reference and hypothesis.
    
#     Args:
#         reference: Original sentence
#         hypothesis: Back-translated sentence
#     Returns:
#         COMET score (float) 
#     """
#     data = [{
#         "src": normalize_text(source),
#         "mt": normalize_text(hypothesis),
#         "ref": normalize_text(reference)
#     }]
#     scores = model.predict(data, batch_size=1, gpus=0, progress_bar=False)
#     return float(scores["system_score"])



# ================================================================================
# Metrics 2.0
# ================================================================================
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class MetricResult:
    """Result from a single evaluation metric."""
    metric_name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


@dataclass
class EvaluationInput:
    """Structured input for evaluation metrics."""
    original_text: str
    back_translation: str
    forward_translation: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, eval_input: EvaluationInput) -> MetricResult:
        """Compute the metric score."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the metric name."""
        pass
    
    @abstractmethod
    def requires_source_text(self) -> bool:
        """Whether this metric needs the source text."""
        pass


class BLEUScore(EvaluationMetric):
    """BLEU score metric for translation quality."""
    
    def __init__(self, max_order: int = 4):
        self.max_order = max_order
    
    def compute(self, eval_input: EvaluationInput) -> MetricResult:
        start_time = datetime.now()
        
        # Placeholder - implement actual BLEU calculation
        from sacrebleu import sentence_bleu
        score = sentence_bleu(eval_input.back_translation, [eval_input.original_text])
        # score = 0.85  # Dummy score
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MetricResult(
            metric_name=self.get_name(),
            score=score.score,
            metadata={"max_order": self.max_order},
            execution_time_ms=execution_time
        )
    
    def get_name(self) -> str:
        return f"BLEU-{self.max_order}"
    
    def requires_source_text(self) -> bool:
        return False  # Only needs reference and hypothesis


class COMETMetric(EvaluationMetric):
    """COMET metric for translation quality."""
    
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        self.model_name = model_name
    
    def compute(self, eval_input: EvaluationInput) -> MetricResult:
        start_time = datetime.now()
        
        # Placeholder - implement actual COMET calculation
        # from comet import download_model, load_from_checkpoint
        # model = load_from_checkpoint(download_model(self.model_name))
        # score = model.predict([{
        #     "src": eval_input.original_text,
        #     "mt": eval_input.forward_translation,
        #     "ref": eval_input.back_translation
        # }])
        score = 0.78  # Dummy score
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MetricResult(
            metric_name=self.get_name(),
            score=score,
            metadata={"model": self.model_name},
            execution_time_ms=execution_time
        )
    
    def get_name(self) -> str:
        return "COMET"
    
    def requires_source_text(self) -> bool:
        return True  # Requires source text


class chrFScore(EvaluationMetric):
    """chrF score metric."""
    
    def compute(self, eval_input: EvaluationInput) -> MetricResult:
        start_time = datetime.now()
        
        # Placeholder - implement actual chrF calculation
        score = 0.82  # Dummy score
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MetricResult(
            metric_name=self.get_name(),
            score=score,
            execution_time_ms=execution_time
        )
    
    def get_name(self) -> str:
        return "chrF"
    
    def requires_source_text(self) -> bool:
        return False


class EditDistanceMetric(EvaluationMetric):
    """Levenshtein edit distance metric."""
    
    def compute(self, eval_input: EvaluationInput) -> MetricResult:
        start_time = datetime.now()
        
        # Placeholder - implement actual edit distance
        distance = 5  # Dummy distance
        max_len = max(len(eval_input.original_text), len(eval_input.back_translation))
        score = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MetricResult(
            metric_name=self.get_name(),
            score=score,
            metadata={"edit_distance": distance},
            execution_time_ms=execution_time
        )
    
    def get_name(self) -> str:
        return "EditDistance"
    
    def requires_source_text(self) -> bool:
        return False
