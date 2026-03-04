from typing import Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from datetime import datetime
from dgp.config import ModelConfig
from dgp.providers import ModelProvider
from dgp.tasks.translation import TranslationPipeline


@dataclass
class TranslationScore:
    """Model-based quality score for a single translation."""
    score: float              # 1.0 – 5.0
    label: str                # e.g. "good", "poor"
    reasoning: str            # one-sentence justification from the model


@dataclass
class SentenceGenMetadata:
    """Metadata for tracking sentence generation pipeline execution."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    provider_name: str = ""
    error: Optional[str] = None


class SentenceGenState(TypedDict):
    """State for sentence generation workflow."""
    # Required fields
    word: str
    model_config: ModelConfig
    generation_provider: ModelProvider
    translation_pipeline: TranslationPipeline

    # Intermediate & final results
    descriptive_word: str                  # Step 1: synonym/descriptor for the original word
    original_sentence: str                 # Step 2a: sentence using the original word
    substituted_sentence: str              # Step 2b: sentence with original word replaced by descriptive_word
    translated_descriptive_word: str       # Step 3: descriptive_word translated to Kinyarwanda
    translated_substituted_sentence: str   # Step 4: substituted_sentence translated to Kinyarwanda

    # Translation quality scores (Step 5)
    score_translated_descriptive_word: Optional[TranslationScore]
    score_translated_substituted_sentence: Optional[TranslationScore]

    # Metadata
    metadata: SentenceGenMetadata


# ============================================================================
# Node Functions
# ============================================================================

def generate_descriptive_word(state: SentenceGenState) -> SentenceGenState:
    """
    Step 1 — Generate a single descriptive word (synonym/descriptor) for the input word.
    e.g. "happy" → "elated"
    """
    try:
        provider = state["generation_provider"]
        config = state["model_config"]

        system = (
            "You are a linguistic assistant. Given a word, return exactly one descriptive "
            "synonym or short phrase (of at most 4 words) that could replace it in a sentence. "
            "Prefer a single word when possible, but use a short phrase when a single word would "
            "create a lexical gap — i.e. when no simple, common equivalent exists. "
            "The word or phrase must be simple, common, and concrete — avoid rare, technical, or "
            "culturally specific terms so that it can be easily and accurately translated into "
            "low-resource languages. "
            "Return only the word or phrase, nothing else."
        )

        descriptive_word = provider.invoke(
            text=state["word"],
            system=system,
            config=config
        )

        state["descriptive_word"] = descriptive_word.strip()

    except Exception as e:
        state["metadata"].error = f"Descriptive word generation failed: {str(e)}"
        raise

    return state


def generate_original_sentence(state: SentenceGenState) -> SentenceGenState:
    """
    Step 2a — Generate a sentence using the original word.
    e.g. "happy" → "She felt happy when she heard the good news."
    """
    try:
        provider = state["generation_provider"]
        config = state["model_config"]

        system = (
            "You are a helpful assistant. Given a word, generate one clear and natural sentence "
            "that uses it meaningfully. Return only the sentence."
        )

        sentence = provider.invoke(
            text=state["word"],
            system=system,
            config=config
        )

        state["original_sentence"] = sentence.strip()

    except Exception as e:
        state["metadata"].error = f"Original sentence generation failed: {str(e)}"
        raise

    return state


def substitute_with_descriptive_word(state: SentenceGenState) -> SentenceGenState:
    """
    Step 2b — Replace the original word in the sentence with the descriptive word,
    ensuring grammatical correctness.
    e.g. "She felt happy when she heard the good news."
         + descriptive_word "elated"
      →  "She felt elated when she heard the good news."
    """
    try:
        provider = state["generation_provider"]
        config = state["model_config"]

        system = (
            "You are a grammar-aware text editor. You will be given a sentence, an original word, "
            "and a replacement word. Replace every occurrence of the original word in the sentence "
            "with the replacement word, adjusting articles (a/an), verb forms, or capitalisation as "
            "needed to keep the sentence grammatically correct. "
            "Return only the final corrected sentence."
        )

        prompt = (
            f"Sentence: {state['original_sentence']}\n"
            f"Original word: {state['word']}\n"
            f"Replacement word: {state['descriptive_word']}"
        )

        substituted = provider.invoke(
            text=prompt,
            system=system,
            config=config
        )

        state["substituted_sentence"] = substituted.strip()

    except Exception as e:
        state["metadata"].error = f"Word substitution failed: {str(e)}"
        raise

    return state


def translate_descriptive_word(state: SentenceGenState) -> SentenceGenState:
    """
    Step 3 — Translate the descriptive word into Kinyarwanda using TranslationPipeline.
    e.g. "elated" → "ibyishimo"
    """
    try:
        translation_result = state["translation_pipeline"].run(
            text=state["descriptive_word"],
            source_lang="English",
            target_lang="Kinyarwanda",
            # system_template=(
            #     "Translate the following word from {src_lang} to {tgt_lang}. "
            #     "Return only the translated word."
            # )
            system_template=(
                "You are a careful translator from {src_lang} to {tgt_lang}, "
                "specializing in low-resource language translation.\n\n"
                "GUIDELINES:\n"
                "  - Use the most common, everyday equivalent in {tgt_lang} — avoid rare or literary terms\n"
                "  - If no single word exists, use a short natural phrase (at most 4 words) rather than forcing an unnatural word\n"
                "  - Preserve the core meaning exactly — do not add, remove, or reinterpret\n"
                "  - Use standard {tgt_lang} orthography and morphology\n"
                "  - Never leave English words untranslated\n\n"
                "Return only the translated word or phrase, nothing else."
            )
        )

        state["translated_descriptive_word"] = translation_result["translation"].strip()

    except Exception as e:
        state["metadata"].error = f"Descriptive word translation failed: {str(e)}"
        raise

    return state


def translate_substituted_sentence(state: SentenceGenState) -> SentenceGenState:
    """
    Step 4 — Translate the substituted sentence into Kinyarwanda using TranslationPipeline,
    anchoring the already-translated descriptive word so the model uses it consistently
    instead of re-translating it independently.
    e.g. descriptive_word="elated", translated_descriptive_word="ibyishimo"
         substituted_sentence="She felt elated when she heard the good news."
      →  "Yumvise ibyishimo igihe yumvise amakuru meza."
    """
    try:
        system_template = (
            "Translate the following sentence from {{src_lang}} to {{tgt_lang}}. "
            "The word \"{descriptive_word}\" has already been translated as \"{translated_descriptive_word}\". "
            "Use \"{translated_descriptive_word}\" consistently in the translation wherever \"{descriptive_word}\" appears. "
            "Return only the translated sentence."
        ).format(
            descriptive_word=state["descriptive_word"],
            translated_descriptive_word=state["translated_descriptive_word"],
        )

        translation_result = state["translation_pipeline"].run(
            text=state["substituted_sentence"],
            source_lang="English",
            target_lang="Kinyarwanda",
            system_template=system_template
        )

        state["translated_substituted_sentence"] = translation_result["translation"].strip()
        state["metadata"].end_time = datetime.now()

    except Exception as e:
        state["metadata"].error = f"Substituted sentence translation failed: {str(e)}"
        raise

    return state


def _parse_score_response(response: str) -> TranslationScore:
    """Parse the model's JSON score response into a TranslationScore."""
    import json
    import re

    # Strip markdown fences if present
    clean = re.sub(r"```json|```", "", response).strip()
    data = json.loads(clean)

    score = float(data.get("score", 0))
    label = "good" if score >= 4.0 else "acceptable" if score >= 2.5 else "poor"

    return TranslationScore(
        score=score,
        label=label,
        reasoning=data.get("reasoning", "").strip()
    )


def score_translations(state: SentenceGenState) -> SentenceGenState:
    """
    Step 5 — Score both translations using a model-based evaluator.

    For each translation the model is asked to rate on three dimensions:
        - Fluency:   Does the Kinyarwanda output read naturally?
        - Adequacy:  Does it preserve the meaning of the English source?
        - Consistency: Is the descriptive word translated consistently?

    Returns a composite score from 1.0 (very poor) to 5.0 (excellent)
    along with a short reasoning string.
    """
    try:
        provider = state["generation_provider"]
        config = state["model_config"]

        system = (
            "You are a strict translation quality evaluator for English to Kinyarwanda. "
            "You are critical by default — a score of 5 must be near-perfect and is rare. "
            "Score each dimension from 1 to 5 using ONLY these definitions:\n\n"

            "FLUENCY (does the Kinyarwanda read naturally to a native speaker?):\n"
            "  5 = Completely natural, indistinguishable from native writing\n"
            "  4 = Mostly natural with at most one minor awkward phrase\n"
            "  3 = Understandable but noticeably non-native in structure or word choice\n"
            "  2 = Difficult to read, multiple unnatural constructions\n"
            "  1 = Unreadable or grammatically broken\n\n"

            "ADEQUACY (is the full meaning of the source preserved?):\n"
            "  5 = All meaning preserved, nothing added or lost\n"
            "  4 = Meaning mostly preserved, one minor omission or imprecision\n"
            "  3 = Core meaning present but noticeable information lost or distorted\n"
            "  2 = Significant meaning lost, only partial content transferred\n"
            "  1 = Meaning is wrong or the translation is unrelated to the source\n\n"

            "CONSISTENCY (are key terms, especially the descriptive word, translated uniformly?):\n"
            "  5 = All key terms translated consistently and correctly throughout\n"
            "  4 = Consistent with one minor term variation that doesn't affect meaning\n"
            "  3 = Some inconsistency in key term translation\n"
            "  2 = Key terms translated differently across the output\n"
            "  1 = Key terms mistranslated or omitted entirely\n\n"
            "Compute a composite score as the average of the three criteria. "
            "Respond ONLY with a JSON object in this exact format, no extra text:\n"
            '{{"score": <float>, "fluency": <float>, "adequacy": <float>, "consistency": <float>, "reasoning": "<one sentence>"}}'
        )

        # Score 1: translated_descriptive_word
        prompt_word = (
            f"Source (English): {state['descriptive_word']}\n"
            f"Translation (Kinyarwanda): {state['translated_descriptive_word']}"
        )
        response_word = provider.invoke(text=prompt_word, system=system, config=config)
        state["score_translated_descriptive_word"] = _parse_score_response(response_word)

        # Score 2: translated_substituted_sentence
        prompt_sentence = (
            f"Source (English): {state['substituted_sentence']}\n"
            f"Translation (Kinyarwanda): {state['translated_substituted_sentence']}"
        )
        response_sentence = provider.invoke(text=prompt_sentence, system=system, config=config)
        state["score_translated_substituted_sentence"] = _parse_score_response(response_sentence)

        state["metadata"].end_time = datetime.now()

    except Exception as e:
        state["metadata"].error = f"Translation scoring failed: {str(e)}"
        raise

    return state


# ============================================================================
# Pipeline
# ============================================================================

class SentenceGenerationPipeline:
    """
    SentenceGenerationPipeline
    --------------------------
    A multi-step LangGraph workflow that takes a single word and runs it through
    four chained tasks:

        1. generate_descriptive_word
           → Produces a descriptive synonym for the input word.
             e.g. "happy" → "elated"

        2. generate_original_sentence
           → Generates a sentence using the original word.
             e.g. "happy" → "She felt happy when she heard the good news."

        3. substitute_with_descriptive_word
           → Replaces the original word in that sentence with the descriptive word,
             preserving grammar.
             e.g. → "She felt elated when she heard the good news."

        4. translate_descriptive_word
           → Translates the descriptive word into Kinyarwanda via TranslationPipeline.
             e.g. "elated" → "ibyishimo"

        5. translate_substituted_sentence
           → Translates the full substituted sentence into Kinyarwanda via TranslationPipeline.
             e.g. "She felt elated when she heard the good news."
                  → "Yumvise ibyishimo igihe yumvise amakuru meza."

        6. score_translations
           → Model-based quality scoring (1.0–5.0) of both translations on
             fluency, adequacy, and consistency.
             e.g. score=4.2, label="good", reasoning="The translation is fluent and preserves meaning."

    Components required:
        - provider: ModelProvider
            Backend for generation steps (OpenAI, Groq, etc.)
        - translation_pipeline: TranslationPipeline
            Pre-configured pipeline used for the Kinyarwanda translation step.
        - model_config: Optional[ModelConfig]
            Model name, temperature, and runtime config for generation steps.

    Usage Example:
    --------------
        from dgp.providers import GroqProvider
        from dgp.tasks.translation import TranslationPipeline
        from dgp.tasks.sentence_generation import SentenceGenerationPipeline

        provider = GroqProvider()

        translation_pipeline = TranslationPipeline(
            provider=provider,
            model_config=ModelConfig(model_name="openai/gpt-oss-120b", temperature=0.0)
        )

        pipeline = SentenceGenerationPipeline(
            provider=provider,
            translation_pipeline=translation_pipeline,
            model_config=ModelConfig(model_name="openai/gpt-oss-120b", temperature=0.7)
        )

        result = pipeline.run(word="happy")
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
        workflow = StateGraph(SentenceGenState)

        # Register nodes
        workflow.add_node("generate_descriptive_word", generate_descriptive_word)
        workflow.add_node("generate_original_sentence", generate_original_sentence)
        workflow.add_node("substitute_with_descriptive_word", substitute_with_descriptive_word)
        workflow.add_node("translate_descriptive_word", translate_descriptive_word)
        workflow.add_node("translate_substituted_sentence", translate_substituted_sentence)
        workflow.add_node("score_translations", score_translations)

        # Chain: step 1 → step 2a → step 2b → step 3 → step 4 → step 5 → END
        workflow.set_entry_point("generate_descriptive_word")
        workflow.add_edge("generate_descriptive_word", "generate_original_sentence")
        workflow.add_edge("generate_original_sentence", "substitute_with_descriptive_word")
        workflow.add_edge("substitute_with_descriptive_word", "translate_descriptive_word")
        workflow.add_edge("translate_descriptive_word", "translate_substituted_sentence")
        workflow.add_edge("translate_substituted_sentence", "score_translations")
        workflow.add_edge("score_translations", END)

        return workflow.compile()

    def run(self, word: str) -> Dict[str, Any]:
        """Execute the full sentence generation pipeline."""

        if not word.strip():
            raise ValueError("Word must not be empty")

        initial_state: SentenceGenState = {
            "word": word.strip(),
            "model_config": self.model_config,
            "generation_provider": self.provider,
            "translation_pipeline": self.translation_pipeline,
            # Results (populated by nodes)
            "descriptive_word": "",
            "original_sentence": "",
            "substituted_sentence": "",
            "translated_descriptive_word": "",
            "translated_substituted_sentence": "",
            "score_translated_descriptive_word": None,
            "score_translated_substituted_sentence": None,
            "metadata": SentenceGenMetadata(
                pipeline_id=f"sg-{datetime.now().timestamp()}",
                start_time=datetime.now(),
                provider_name=self.provider.get_provider_name()
            )
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "word": final_state["word"],
            "descriptive_word": final_state["descriptive_word"],
            "original_sentence": final_state["original_sentence"],
            "substituted_sentence": final_state["substituted_sentence"],
            "translated_descriptive_word": final_state["translated_descriptive_word"],
            "translated_substituted_sentence": final_state["translated_substituted_sentence"],
            "scores": {
                "translated_descriptive_word": {
                    "score": final_state["score_translated_descriptive_word"].score,
                    "label": final_state["score_translated_descriptive_word"].label,
                    "reasoning": final_state["score_translated_descriptive_word"].reasoning,
                } if final_state["score_translated_descriptive_word"] else None,
                "translated_substituted_sentence": {
                    "score": final_state["score_translated_substituted_sentence"].score,
                    "label": final_state["score_translated_substituted_sentence"].label,
                    "reasoning": final_state["score_translated_substituted_sentence"].reasoning,
                } if final_state["score_translated_substituted_sentence"] else None,
            },
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
    from dgp.providers import GroqProvider
    from dgp.tasks.translation import TranslationPipeline

    provider = GroqProvider()

    translation_pipeline = TranslationPipeline(
        provider=provider,
        model_config=ModelConfig(model_name="openai/gpt-oss-20b", temperature=0.0)
    )

    pipeline = SentenceGenerationPipeline(
        provider=provider,
        translation_pipeline=translation_pipeline,
        model_config=ModelConfig(model_name="openai/gpt-oss-20b", temperature=0.7)
    )

    result = pipeline.run(word="nostalgia")

    # Example output:
    # {
    #   "word": "happy",
    #   "descriptive_word": "elated",
    #   "original_sentence": "She felt happy when she heard the good news.",
    #   "substituted_sentence": "She felt elated when she heard the good news.",
    #   "translated_descriptive_word": "ibyishimo",
    #   "translated_substituted_sentence": "Yumvise ibyishimo igihe yumvise amakuru meza.",
    #   "scores": {
    #     "translated_descriptive_word": {"score": 4.3, "label": "good", "reasoning": "..."},
    #     "translated_substituted_sentence": {"score": 4.1, "label": "good", "reasoning": "..."},
    #   },
    #   "metadata": { ... }
    # }
    print(result)