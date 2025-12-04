from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from langgraph.graph import StateGraph, END
from datetime import datetime
from dgp.config import ModelConfig
from dgp.providers import ModelProvider
from dgp.metrics import EvaluationMetric, EvaluationInput, MetricResult

@dataclass
class BackTranslationMetadata:
    """Metadata for tracking translation pipeline execution."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    provider_name: str = ""
    error: Optional[str] = None


from typing import TypedDict, Annotated
from operator import add

class BackTranslationState(TypedDict):
    """State for back-translation workflow."""
    # Required fields
    original_text: str
    source_lang: str
    intermediate_lang: str
    system_template: str = "Translate from {src_lang} to {tgt_lang}"
    model_config: ModelConfig
    translation_provider: ModelProvider
    evaluation_metrics: List[EvaluationMetric]
    
    # Translation results
    forward_translation: str
    back_translation: str
    
    # Evaluation results
    evaluation_input: Optional[EvaluationInput]
    metric_results: List[MetricResult]
    overall_score: float
    
    # Metadata
    metadata: BackTranslationMetadata


# ============================================================================
# Node Functions
# ============================================================================

def forward_translate(state: BackTranslationState) -> BackTranslationState:
    """Translate from source language to intermediate language."""
    try:
        provider = state["translation_provider"]
        system = state["system_template"].format(
            src_lang=state["source_lang"],
            tgt_lang=state["intermediate_lang"]
        )
        config = state["model_config"]
        
        translation = provider.invoke(
            text=state["original_text"],
            system=system,
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
        system = state["system_template"].format(
            src_lang=state["intermediate_lang"],
            tgt_lang=state["source_lang"]
        )
        
        translation = provider.invoke(
            text=state["forward_translation"],
            system=system,
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

class BackTranslationPipeline:
    """Builder for back-translation workflows."""
    
    def __init__(
        self,
        provider: ModelProvider,
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
        intermediate_lang: str,
        system_template: str,
    ) -> Dict[str, Any]:
        """Execute the back-translation pipeline."""
        
        # Validate language pair
        if source_lang == intermediate_lang:
            raise ValueError("Source and intermediate languages must be different")
        
        # Initialize state as a proper dict
        initial_state: BackTranslationState = {
            "original_text": text,
            "source_lang": source_lang,
            "intermediate_lang": intermediate_lang,
            "system_template": system_template,
            "model_config": self.model_config,
            "translation_provider": self.provider,
            "evaluation_metrics": self.metrics,
            "forward_translation": "",
            "back_translation": "",
            "evaluation_input": None,
            "metric_results": [],
            "overall_score": 0.0,
            "metadata": BackTranslationMetadata(
                pipeline_id=f"bt-{datetime.now().timestamp()}",
                start_time=datetime.now(),
                provider_name=self.provider.get_provider_name()
            )
        }
        
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

if __name__ == "__main__":
    # Example 1: OpenAI with BLEU and COMET
    from dgp.providers import GroqProvider
    from dgp.metrics import BLEUScore, COMETMetric
    src_lang: str = "English"
    tgt_lang: str = "French"
    pipeline1 = BackTranslationPipeline(
        provider=GroqProvider(),
        metrics=[
            BLEUScore(max_order=4), 
            # COMETMetric()
            ],
        model_config=ModelConfig(
            model_name="openai/gpt-oss-120b", 
            temperature=0.0,
        )
    )
    
    result1 = pipeline1.run(
        text="Hello, how are you?",
        source_lang="English",
        intermediate_lang="Kinyarwanda",
        system_template="Translate the following text from {src_lang} to {tgt_lang}"
    )
    print("Result 1:", result1)