from typing import Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from datetime import datetime
from dgp.config import ModelConfig
from dgp.providers import ModelProvider
from dgp.tasks.translation import TranslationPipeline


@dataclass
class BaseGenMetadata:
    """Metadata for tracking base generation pipeline execution."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    provider_name: str = ""
    error: Optional[str] = None


class BaseGenState(TypedDict):
    """State for base word → sentence → translation workflow."""
    # Required fields
    word: str
    model_config: ModelConfig
    generation_provider: ModelProvider
    translation_pipeline: TranslationPipeline

    # Results
    sentence: str             # Step 1: sentence generated from the word
    translation: str          # Step 2: Kinyarwanda translation of the sentence

    # Metadata
    metadata: BaseGenMetadata


# ============================================================================
# Node Functions
# ============================================================================

def generate_sentence(state: BaseGenState) -> BaseGenState:
    """
    Step 1 — Generate a simple, translation-friendly sentence from the input word.
    e.g. "rain" → "The rain fell softly on the dry ground."
    """
    try:
        provider = state["generation_provider"]
        config = state["model_config"]

        system = (
            "You are a helpful assistant. Given a word, generate one short and simple sentence "
            "that uses it meaningfully. "
            "Follow these rules to ensure the sentence is easy to translate into low-resource languages:\n"
            "- Use common, everyday vocabulary — avoid idioms, slang, or figurative language\n"
            "- Keep the sentence short (under 15 words)\n"
            "- Use simple sentence structure (Subject + Verb + Object) — avoid subordinate clauses\n"
            "- Avoid cultural references, proper nouns, or region-specific expressions\n"
            "- Use active voice\n"
            "Return only the sentence."
        )

        sentence = provider.invoke(
            text=state["word"],
            system=system,
            config=config
        )

        state["sentence"] = sentence.strip()

    except Exception as e:
        state["metadata"].error = f"Sentence generation failed: {str(e)}"
        raise

    return state


def translate_sentence(state: BaseGenState) -> BaseGenState:
    """
    Step 2 — Translate the generated sentence into Kinyarwanda using TranslationPipeline.
    e.g. "The rain fell softly on the dry ground."
      →  "Imvura yarashe neza ku butaka gukaba."
    """
    try:
        system_template = (
            "You are a careful translator from {{src_lang}} to {{tgt_lang}}, "
            "specializing in low-resource language translation.\n\n"
            "The key word in this sentence is \"{word}\".\n"
            "If \"{word}\" does not have a direct, natural equivalent in {{tgt_lang}}, "
            "do not leave it untranslated or force a borrowed term. Instead, replace it "
            "with the simplest synonym or short phrase (at most 4 words) that preserves "
            "its meaning and translates cleanly into {{tgt_lang}}.\n\n"
            "GUIDELINES:\n"
            "  - Use simple, natural {{tgt_lang}} sentence structure — do not mirror English word order\n"
            "  - Use common vocabulary only — avoid rare, technical, or borrowed terms unless no alternative exists\n"
            "  - Preserve the full meaning of the source — do not add, omit, or distort any content\n"
            "  - Use standard {{tgt_lang}} orthography and morphology\n"
            "  - Never leave English words untranslated\n\n"
            "Return only the translated sentence."
        ).format(word=state["word"])

        translation_result = state["translation_pipeline"].run(
            text=state["sentence"],
            source_lang="English",
            target_lang="Kinyarwanda",
            system_template=system_template
        )

        state["translation"] = translation_result["translation"].strip()
        state["metadata"].end_time = datetime.now()

    except Exception as e:
        state["metadata"].error = f"Translation failed: {str(e)}"
        raise

    return state


# ============================================================================
# Pipeline
# ============================================================================

class BaseGenerationPipeline:
    """
    BaseGenerationPipeline
    ----------------------
    A minimal two-step LangGraph workflow: word → sentence → Kinyarwanda translation.

        1. generate_sentence
           → Generates a simple, translation-friendly sentence from the input word.
             e.g. "rain" → "The rain fell softly on the dry ground."

        2. translate_sentence
           → Translates the sentence into Kinyarwanda via TranslationPipeline.
             e.g. → "Imvura yarashe neza ku butaka gukaba."

    Components required:
        - provider: ModelProvider
            Backend for the sentence generation step (OpenAI, Groq, etc.)
        - translation_pipeline: TranslationPipeline
            Pre-configured pipeline used for the translation step.
        - model_config: Optional[ModelConfig]
            Model name, temperature, and runtime config for generation.

    Usage Example:
    --------------
        from dgp.providers import GroqProvider
        from dgp.tasks.translation import TranslationPipeline
        from dgp.tasks.base_generation import BaseGenerationPipeline

        provider = GroqProvider()

        translation_pipeline = TranslationPipeline(
            provider=provider,
            model_config=ModelConfig(model_name="openai/gpt-oss-120b", temperature=0.0)
        )

        pipeline = BaseGenerationPipeline(
            provider=provider,
            translation_pipeline=translation_pipeline,
            model_config=ModelConfig(model_name="openai/gpt-oss-120b", temperature=0.7)
        )

        result = pipeline.run(word="rain")
        print(result)
    """

    def __init__(
        self,
        provider: ModelProvider,
        translation_pipeline: TranslationPipeline,
        model_config: Optional[ModelConfig] = None,
    ):
        self.provider = provider
        self.translation_pipeline = translation_pipeline
        self.model_config = model_config or ModelConfig(
            model_name="default",
            temperature=0.7
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(BaseGenState)

        workflow.add_node("generate_sentence", generate_sentence)
        workflow.add_node("translate_sentence", translate_sentence)

        workflow.set_entry_point("generate_sentence")
        workflow.add_edge("generate_sentence", "translate_sentence")
        workflow.add_edge("translate_sentence", END)

        return workflow.compile()

    def run(self, word: str) -> Dict[str, Any]:
        """Execute the base generation pipeline."""

        if not word.strip():
            raise ValueError("Word must not be empty")

        initial_state: BaseGenState = {
            "word": word.strip(),
            "model_config": self.model_config,
            "generation_provider": self.provider,
            "translation_pipeline": self.translation_pipeline,
            "sentence": "",
            "translation": "",
            "metadata": BaseGenMetadata(
                pipeline_id=f"bg-{datetime.now().timestamp()}",
                start_time=datetime.now(),
                provider_name=self.provider.get_provider_name()
            )
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "word": final_state["word"],
            "sentence": final_state["sentence"],
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
if __name__ == "__main__":
    from dgp.providers import GroqProvider, VLLMProvider

    # provider = GroqProvider()
    provider = VLLMProvider(
        base_url="http://localhost:10000/v1"
    )

    translation_pipeline = TranslationPipeline(
        provider=provider,
        model_config=ModelConfig(
            model_name="openai/gpt-oss-20b", 
            temperature=0.0,
            max_tokens=1024
        )
    )

    pipeline = BaseGenerationPipeline(
        provider=provider,
        translation_pipeline=translation_pipeline,
        model_config=ModelConfig(
            model_name="openai/gpt-oss-20b", 
            temperature=0.7,
            max_tokens=1024
            )
    )

    result = pipeline.run(word="rain")

    # Example output:
    # {
    #   "word": "rain",
    #   "sentence": "The rain fell softly on the dry ground.",
    #   "translation": "Imvura yarashe neza ku butaka gukaba.",
    #   "metadata": { ... }
    # }
    print(result)