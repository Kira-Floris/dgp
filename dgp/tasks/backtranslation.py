from typing import (
    TypedDict, 
    Callable, 
    Dict,
    List, 
    Any, 
    Optional, 
    runtime_checkable, 
    Protocol
)
from langgraph.graph import StateGraph, END
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.0
    system_prompt: str = ""
    max_tokens: Optional[int] = None

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")

@runtime_checkable
class TranslationProvider(Protocol):
    def translate(
            self,
            text: str,
            source_lang: str,
            target_lang: str,
            config: ModelConfig
    ) -> str:
        ...
    
    def get_provider_name(self) -> str:
        ...

class OpenAITranslator:
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        config: ModelConfig
    ) -> str:
        # Placeholder - implement actual OpenAI API call
        prompt = f"Translate the following text from {source_lang.value} to {target_lang.value}:\n\n{text}"
        # response = openai.ChatCompletion.create(...)
        return f"[OpenAI Translation: {text}]"
    
    def get_provider_name(self) -> str:
        return "OpenAI"
    
@dataclass
class MetricResult:
    metric_name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

@dataclass
class EvaluationInput:
    original_text: str
    forward_translation: str
    back_translation: str
    source_lang: str
    target_lang: str

class EvaluationMetric(ABC):
    @abstractmethod
    def compute(self, eval_input: EvaluationInput) -> MetricResult:
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
        # from sacrebleu import sentence_bleu
        # score = sentence_bleu(eval_input.back_translation, [eval_input.original_text])
        score = 0.85  # Dummy score
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MetricResult(
            metric_name=self.get_name(),
            score=score,
            metadata={"max_order": self.max_order},
            execution_time_ms=execution_time
        )
    
    def get_name(self) -> str:
        return f"BLEU-{self.max_order}"
    
    def requires_source_text(self) -> bool:
        return False  # Only needs reference and hypothesis



class BackTranslationMetadata:
    """Metadata for tracking translation pipeline execution."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    provider_name: str = ""
    error: Optional[str] = None


class BackTranslationState(Dict[str, Any]):
    """State for back-translation workflow."""
    
    def __init__(
        self,
        original_text: str,
        source_lang: str,
        intermediate_lang: str,
        model_config: ModelConfig,
        translation_provider: TranslationProvider,
        evaluation_metrics: List[EvaluationMetric],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Required fields
        self["original_text"] = original_text
        self["source_lang"] = source_lang
        self["intermediate_lang"] = intermediate_lang
        self["model_config"] = model_config
        self["translation_provider"] = translation_provider
        self["evaluation_metrics"] = evaluation_metrics
        
        # Translation results
        self["forward_translation"] = ""
        self["back_translation"] = ""
        
        # Evaluation results
        self["evaluation_input"] = None
        self["metric_results"] = []
        self["overall_score"] = 0.0
        
        # Metadata
        self["metadata"] = BackTranslationMetadata(
            pipeline_id=f"bt-{datetime.now().timestamp()}",
            start_time=datetime.now(),
            provider_name=translation_provider.get_provider_name()
        )

def forward_translate(state: BackTranslationState) -> BackTranslationState:
    """Translate from source language to intermediate language."""
    try:
        provider = state["translation_provider"]
        config = state["model_config"]
        
        translation = provider.translate(
            text=state["original_text"],
            source_lang=state["source_lang"],
            target_lang=state["intermediate_lang"],
            config=config
        )
        
        state["forward_translation"] = translation
        
    except Exception as e:
        state["metadata"].error = f"Forward translation failed: {str(e)}"
        raise
    
    return state


def backward_translate(state: BackTranslationState) -> BackTranslationState:
    """Translate from intermediate language back to source language."""
    try:
        provider = state["translation_provider"]
        config = state["model_config"]
        
        translation = provider.translate(
            text=state["forward_translation"],
            source_lang=state["intermediate_lang"],
            target_lang=state["source_lang"],
            config=config
        )
        
        state["back_translation"] = translation
        
    except Exception as e:
        state["metadata"].error = f"Backward translation failed: {str(e)}"
        raise
    
    return state


def evaluate_backtranslation(state: BackTranslationState) -> BackTranslationState:
    """Evaluate translation quality using multiple metrics."""
    try:
        # Prepare evaluation input
        eval_input = EvaluationInput(
            original_text=state["original_text"],
            back_translation=state["back_translation"],
            forward_translation=state["forward_translation"],
            source_lang=state["source_lang"],
            target_lang=state["intermediate_lang"]
        )
        state["evaluation_input"] = eval_input
        
        # Run all metrics
        metric_results = []
        for metric in state["evaluation_metrics"]:
            result = metric.compute(eval_input)
            metric_results.append(result)
        
        state["metric_results"] = metric_results
        
        # Calculate overall score (weighted average)
        if metric_results:
            state["overall_score"] = sum(r.score for r in metric_results) / len(metric_results)
        
        # Update metadata
        state["metadata"].end_time = datetime.now()
        
    except Exception as e:
        state["metadata"].error = f"Evaluation failed: {str(e)}"
        raise
    
    return state


# ============================================================================
# Pipeline Builder
# ============================================================================

class BackTranslationPipeline:
    """Builder for back-translation workflows."""
    
    def __init__(
        self,
        provider: TranslationProvider,
        metrics: List[EvaluationMetric],
        model_config: Optional[ModelConfig] = None
    ):
        self.provider = provider
        self.metrics = metrics
        self.model_config = model_config or ModelConfig(
            model_name="default",
            temperature=0.0
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(BackTranslationState)
        
        # Add nodes
        workflow.add_node("forward_translate", forward_translate)
        workflow.add_node("backward_translate", backward_translate)
        workflow.add_node("evaluate", evaluate_backtranslation)
        
        # Define edges
        workflow.set_entry_point("forward_translate")
        workflow.add_edge("forward_translate", "backward_translate")
        workflow.add_edge("backward_translate", "evaluate")
        workflow.add_edge("evaluate", END)
        
        return workflow.compile()
    
    def run(
        self,
        text: str,
        source_lang: str,
        intermediate_lang: str
    ) -> Dict[str, Any]:
        """Execute the back-translation pipeline."""
        
        # Validate language pair
        if source_lang == intermediate_lang:
            raise ValueError("Source and intermediate languages must be different")
        
        # Initialize state
        initial_state = BackTranslationState(
            original_text=text,
            source_lang=source_lang,
            intermediate_lang=intermediate_lang,
            model_config=self.model_config,
            translation_provider=self.provider,
            evaluation_metrics=self.metrics
        )
        
        # Run pipeline
        final_state = self.graph.invoke(initial_state)
        
        return {
            "original": final_state["original_text"],
            "forward": final_state["forward_translation"],
            "back": final_state["back_translation"],
            "metrics": [
                {
                    "name": r.metric_name,
                    "score": r.score,
                    "metadata": r.metadata,
                    "time_ms": r.execution_time_ms
                }
                for r in final_state["metric_results"]
            ],
            "overall_score": final_state["overall_score"],
            "metadata": {
                "pipeline_id": final_state["metadata"].pipeline_id,
                "provider": final_state["metadata"].provider_name,
                "duration_ms": (
                    (final_state["metadata"].end_time - final_state["metadata"].start_time).total_seconds() * 1000
                    if final_state["metadata"].end_time else 0
                )
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: OpenAI with BLEU and COMET
    pipeline1 = BackTranslationPipeline(
        provider=OpenAITranslator(),
        metrics=[BLEUScore(max_order=4)],
        model_config=ModelConfig(model_name="gpt-4", temperature=0.0)
    )
    
    result1 = pipeline1.run(
        text="Hello, how are you?",
        source_lang="English",
        intermediate_lang="French"
    )
    print("Result 1:", result1)
    