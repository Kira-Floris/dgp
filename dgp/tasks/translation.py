from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from datetime import datetime
from dgp.config import ModelConfig
from dgp.providers import ModelProvider


@dataclass
class TranslationMetadata:
    """Metadata for tracking translation pipeline execution."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    provider_name: str = ""
    error: Optional[str] = None

class TranslationState(TypedDict):
    """State for translation workflow."""
    # Required fields
    original_text: str
    source_lang: str
    target_lang: str
    system_template: str
    model_config: ModelConfig
    translation_provider: ModelProvider
    
    # Translation result
    translation: str
    
    # Metadata
    metadata: TranslationMetadata


# ============================================================================
# Node Functions
# ============================================================================

def translate(state: TranslationState) -> TranslationState:
    """Translate from source language to target language."""
    try:
        provider = state["translation_provider"]
        system = state["system_template"].format(
            src_lang=state["source_lang"],
            tgt_lang=state["target_lang"]
        )
        config = state["model_config"]
        
        translation = provider.invoke(
            text=state["original_text"],
            system=system,
            config=config
        )
        
        state["translation"] = translation
        state["metadata"].end_time = datetime.now()
        
    except Exception as e:
        state["metadata"].error = f"Translation failed: {str(e)}"
        raise
    
    return state


class TranslationPipeline:
    """
    TranslationPipeline
    -------------------
    A lightweight workflow builder for performing single-direction translation using LangGraph.

    This pipeline performs:
        1. Translation (source_lang → target_lang)

    Components required:
        - provider: ModelProvider
            Backend used to run the translation (OpenAI, Groq, etc.)
        - model_config: Optional[ModelConfig]
            Defines the model name, temperature, and runtime configuration.

    Usage Example:
    --------------
    Example of how this class is typically run:

        from dgp.providers import GroqProvider
        from dgp.tasks.translation import TranslationPipeline

        pipeline = TranslationPipeline(
            provider=GroqProvider(),
            model_config=ModelConfig(
                model_name="openai/gpt-oss-120b",
                temperature=0.0
            )
        )

        result = pipeline.run(
            text="Amakuru yawe?",
            source_lang="Kinyarwanda",
            target_lang="English",
            system_template="Translate the text from {src_lang} to {tgt_lang}."
        )

        print(result)
    """
    
    def __init__(
        self,
        provider: ModelProvider,
        model_config: Optional[ModelConfig] = None
    ):
        self.provider = provider
        self.model_config = model_config or ModelConfig(
            model_name="default",
            temperature=0.0
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(TranslationState)
        
        # Add node
        workflow.add_node("translate", translate)
        
        # Define edges
        workflow.set_entry_point("translate")
        workflow.add_edge("translate", END)
        
        return workflow.compile()
    
    def run(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        system_template: str = "Translate from {src_lang} to {tgt_lang}",
    ) -> Dict[str, Any]:
        """Execute the translation pipeline."""
        
        # Validate language pair
        if source_lang == target_lang:
            raise ValueError("Source and target languages must be different")
        
        # Initialize state
        initial_state: TranslationState = {
            "original_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "system_template": system_template,
            "model_config": self.model_config,
            "translation_provider": self.provider,
            "translation": "",
            "metadata": TranslationMetadata(
                pipeline_id=f"tr-{datetime.now().timestamp()}",
                start_time=datetime.now(),
                provider_name=self.provider.get_provider_name()
            )
        }
        
        # Run pipeline
        final_state = self.graph.invoke(initial_state)
        
        return {
            "original": final_state["original_text"],
            "translation": final_state["translation"],
            "metadata": {
                "pipeline_id": final_state["metadata"].pipeline_id,
                "provider": final_state["metadata"].provider_name,
                "duration_ms": (
                    (final_state["metadata"].end_time - final_state["metadata"].start_time).total_seconds() * 1000
                    if final_state["metadata"].end_time else 0
                )
            }
        }


# Example usage:
# if __name__ == "__main__":
#     from dgp.providers import GroqProvider
    
#     pipeline = TranslationPipeline(
#         provider=GroqProvider(),
#         model_config=ModelConfig(
#             model_name="openai/gpt-oss-120b",
#             temperature=0.0
#         )
#     )
    
#     result = pipeline.run(
#         text="Amakuru yanyu?",
#         source_lang="Kinyarwanda",
#         target_lang="English",
#         system_template="Translate the following text from {src_lang} to {tgt_lang}. Return the translation only."
#     )
    
#     print("Translation Result:", result)