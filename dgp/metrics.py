import sacrebleu
import string
from comet import load_from_checkpoint
from huggingface_hub import snapshot_download
import shutil
import os

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
import string
import time
from comet import load_from_checkpoint, download_model
from typing import Optional

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


def normalize_text(text: str) -> str:
    """Lowercase + remove punctuation for stable scoring."""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


class COMETMetric(EvaluationMetric):
    """
    Simple COMET metric implementation.
    Automatically uses:
        - English model: Unbabel/wmt22-comet-da
        - Kinyarwanda model: ./comet_model/checkpoints/KinyCOMET+Unbabel.ckpt
    """

    def __init__(
        self,
        eng_model_ckpt: str = "Unbabel/wmt22-comet-da",
        # kin_model_ckpt: Optional[str] = "./comet_model/checkpoints/KinyCOMET+Unbabel.ckpt",
        kin_model_ckpt: str = "chrismazii/kinycomet_unbabel",
        lang: str = "english",
    ):
        self.eng_model_ckpt = eng_model_ckpt
        self.kin_model_ckpt = kin_model_ckpt
        self.lang = lang

        # Load models once
        try:
            self.eng_model = load_from_checkpoint(
                download_model(eng_model_ckpt)
            )
        except Exception as e:
            self.eng_model = None
            raise RuntimeError(f"COMET Model Loading Error: {e}")

        try:
            ckpt_dir = os.path.dirname(download_model(kin_model_ckpt))
            os.makedirs(os.path.join(ckpt_dir, "checkpoints"), exist_ok=True)
            shutil.copy(
                os.path.join(ckpt_dir, "KinyCOMET+Unbabel.ckpt"), 
                os.path.join(ckpt_dir, "checkpoints/KinyCOMET+Unbabel.ckpt")
            )
            shutil.copy(
                os.path.join(ckpt_dir, "KinyCOMET+XLM-Roberta.ckpt"), 
                os.path.join(ckpt_dir, "checkpoints/KinyCOMET+XLM-Roberta.ckpt")
            )
            self.kin_model = load_from_checkpoint(os.path.join(ckpt_dir, "KinyCOMET+Unbabel.ckpt"))
        except Exception as e:
            self.kin_model = None
            raise RuntimeError(f"COMET Model Loading Error: {e}")

    def get_model(self, lang: Optional[str]):
        """Choose English or Kinyarwanda COMET model."""
        if lang and lang.lower() in ["kin", "rw", "kinyarwanda", "Kinyarwanda"] and self.kin_model:
            return self.kin_model
        return self.eng_model

    def compute(self, eval_input: EvaluationInput) -> MetricResult:
        start_t = time.time()

        model = self.get_model(self.lang)

        if model is None:
            raise RuntimeError("COMET model could not be loaded.")

        data = [{
            "src": normalize_text(eval_input.original_text),
            "mt": normalize_text(eval_input.back_translation),
            "ref": normalize_text(eval_input.forward_translation)
        }]

        scores = model.predict(data, batch_size=1, gpus=0, progress_bar=False)
        comet_score = float(scores["system_score"])

        elapsed_ms = (time.time() - start_t) * 1000

        return MetricResult(
            metric_name=self.get_name(),
            score=comet_score,
            metadata={
                "language": eval_input.source_lang,
                "model_used": model.__class__.__name__
            },
            execution_time_ms=elapsed_ms
        )

    def get_name(self) -> str:
        return "COMET"

    def requires_source_text(self) -> bool:
        return True


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
