# """
# generate_training_data.py
# -------------------------
# Batch script that iterates over words from common_df, runs the
# SentenceGenerationPipeline N times per word, and saves results to CSV.

# Failure isolation strategy:
#   - A single sentence generation failure does not skip the word — the
#     remaining 4 attempts still run.
#   - A complete word failure (all 5 attempts fail) is logged and skipped —
#     the rest of the words still run.
#   - All errors are recorded in the CSV row so nothing is silently lost.
#   - Progress is checkpointed after every word so a crash mid-run can be
#     resumed from the last saved row.
# """

# import json
# import logging
# import traceback
# from pathlib import Path

# import pandas as pd
# from tqdm import tqdm

# from dgp.config import ModelConfig
# from dgp.tasks.sentencegeneration import SentenceGenerationPipeline
# from dgp.tasks.translation import TranslationPipeline
# from dgp.providers import ModelProvider, VLLMProvider, GroqProvider

# # ============================================================================
# # Configuration
# # ============================================================================

# OUTPUT_PATH = Path("training_data.csv")
# SENTENCES_PER_WORD = 5
# LOG_PATH = Path("generation_errors.log")

# # ============================================================================
# # Logging
# # ============================================================================

# logging.basicConfig(
#     filename=LOG_PATH,
#     level=logging.ERROR,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )
# logger = logging.getLogger(__name__)

# # ============================================================================
# # CSV schema helpers
# # ============================================================================

# CSV_COLUMNS = [
#     "index",
#     "word",
#     "english_sentence",
#     "kinyarwanda_word_translation",
#     "kinyarwanda_sentence_translation",
#     "substituted_word",
#     "substituted_sentence",
#     "kinyarwanda_word_score",
#     "kinyarwanda_sentence_score",
#     "full_output",
#     "error",
# ]


# def result_to_row(idx: int, word: str, result: dict) -> dict:
#     """Map a pipeline result dict to a flat CSV row."""
#     scores = result.get("scores", {})
#     word_score = scores.get("translated_descriptive_word") or {}
#     sentence_score = scores.get("translated_substituted_sentence") or {}

#     return {
#         "index": idx,
#         "word": word,
#         "english_sentence": result.get("original_sentence", ""),
#         "kinyarwanda_word_translation": result.get("translated_descriptive_word", ""),
#         "kinyarwanda_sentence_translation": result.get("translated_substituted_sentence", ""),
#         "substituted_word": result.get("descriptive_word", ""),
#         "substituted_sentence": result.get("substituted_sentence", ""),
#         "kinyarwanda_word_score": word_score.get("score", ""),
#         "kinyarwanda_sentence_score": sentence_score.get("score", ""),
#         "full_output": json.dumps(result, ensure_ascii=False),
#         "error": "",
#     }


# def error_row(idx: int, word: str, error: str) -> dict:
#     """Build a placeholder CSV row for a failed generation."""
#     return {
#         "index": idx,
#         "word": word,
#         "english_sentence": "",
#         "kinyarwanda_word_translation": "",
#         "kinyarwanda_sentence_translation": "",
#         "substituted_word": "",
#         "substituted_sentence": "",
#         "kinyarwanda_word_score": "",
#         "kinyarwanda_sentence_score": "",
#         "full_output": "",
#         "error": error,
#     }


# # ============================================================================
# # Resume logic
# # ============================================================================

# def get_completed_indices(output_path: Path) -> set:
#     """Return the set of (word, sentence_num) pairs already saved to CSV."""
#     if not output_path.exists():
#         return set()
#     try:
#         existing = pd.read_csv(output_path, usecols=["index"])
#         return set(existing["index"].tolist())
#     except Exception:
#         return set()


# # ============================================================================
# # Pipeline setup
# # ============================================================================

# def build_pipeline(
#     provider: ModelProvider,
#     generation_model_config: ModelConfig,
#     translation_model_config: ModelConfig,
# ) -> SentenceGenerationPipeline:
#     """
#     Build the SentenceGenerationPipeline with explicit provider and model configs.

#     Args:
#         provider:                 The ModelProvider backend to use for all steps.
#         generation_model_config:  ModelConfig for generation nodes (descriptive word,
#                                   sentence generation, substitution, scoring).
#         translation_model_config: ModelConfig for TranslationPipeline nodes
#                                   (word and sentence translation).
#     """
#     translation_pipeline = TranslationPipeline(
#         provider=provider,
#         model_config=translation_model_config,
#     )

#     return SentenceGenerationPipeline(
#         provider=provider,
#         translation_pipeline=translation_pipeline,
#         model_config=generation_model_config,
#     )


# # ============================================================================
# # Main batch loop
# # ============================================================================

# def run_batch(common_df: pd.DataFrame, pipeline: SentenceGenerationPipeline, n_sentences: int = SENTENCES_PER_WORD):
#     completed_indices = get_completed_indices(OUTPUT_PATH)

#     # Write header only if starting fresh
#     write_header = not OUTPUT_PATH.exists()

#     global_idx = 0  # unique row index across all words and sentences

#     words = common_df["word"].tolist()

#     for word in tqdm(words, desc="Words"):
#         word_failures = 0

#         for attempt in range(n_sentences):
#             row_idx = global_idx

#             # Skip already completed rows (resume support)
#             if row_idx in completed_indices:
#                 global_idx += 1
#                 continue

#             try:
#                 result = pipeline.run(word=word)
#                 row = result_to_row(row_idx, word, result)

#             except Exception as e:
#                 word_failures += 1
#                 error_msg = f"word='{word}' attempt={attempt} | {type(e).__name__}: {str(e)}"
#                 logger.error(error_msg + "\n" + traceback.format_exc())
#                 row = error_row(row_idx, word, error_msg)

#             # Checkpoint: flush each sentence to disk immediately after generation
#             pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(
#                 OUTPUT_PATH,
#                 mode="a",
#                 header=write_header,
#                 index=False,
#                 encoding="utf-8-sig",
#             )
#             write_header = False
#             global_idx += 1

#         # Log if all attempts for this word failed
#         if word_failures == n_sentences:
#             logger.error(f"All {n_sentences} attempts failed for word='{word}' — skipping word entirely.")

#     print(f"\nDone. Results saved to: {OUTPUT_PATH}")
#     print(f"Error log: {LOG_PATH}")


# # ============================================================================
# # Entry point
# # ============================================================================

# if __name__ == "__main__":
#     # --- Build common_df (paste your existing notebook cells here) ----------
#     from wordfreq import get_frequency_list, zipf_frequency
#     import spacy

#     nlp = spacy.load("en_core_web_sm")
#     stopwords = nlp.Defaults.stop_words

#     words_list = []
#     commonality = []

#     for words in get_frequency_list("en"):
#         if len(words) < 1:
#             continue
#         for word in words:
#             if word in stopwords:
#                 continue
#             if not word.isalpha():
#                 continue
#             words_list.append(word)
#             commonality.append(zipf_frequency(word, "en"))

#     common_df = pd.DataFrame({
#         "word": words_list,
#         "commonality": commonality
#     }).sort_values("commonality")

#     common_df = common_df[(common_df["commonality"]>3) & (common_df["commonality"]<4)]

#     common_df = common_df.drop_duplicates(subset=["word"]).dropna()
#     common_df["is_alpha"] = common_df["word"].str.isalpha()
#     common_df = common_df[common_df["is_alpha"] == True]
#     common_df = common_df.head(2)
#     # -------------------------------------------------------------------------

#     # -------------------------------------------------------------------------
#     # Provider and model configuration
#     # -------------------------------------------------------------------------

#     # Provider — swap out for OpenAIProvider, AnthropicProvider, etc. as needed
#     # provider = GroqProvider()
#     provider = VLLMProvider(
#         base_url="http://localhost:10000/v1"
#     )

#     # Generation model: used for descriptive word, sentence generation,
#     # substitution, and translation scoring nodes
#     generation_model_config = ModelConfig(
#         model_name="openai/gpt-oss-120b",
#         temperature=1.0,
#     )

#     # Translation model: used for word and sentence translation nodes
#     # Lower temperature for more deterministic translations
#     translation_model_config = ModelConfig(
#         model_name="openai/gpt-oss-120b",
#         temperature=1.0,
#     )

#     # -------------------------------------------------------------------------
#     # Build pipeline and run
#     # -------------------------------------------------------------------------

#     pipeline = build_pipeline(
#         provider=provider,
#         generation_model_config=generation_model_config,
#         translation_model_config=translation_model_config,
#     )

#     run_batch(common_df, pipeline=pipeline)



"""
generate_training_data.py
-------------------------
Batch script that iterates over words from common_df, runs the
SentenceGenerationPipeline N times per word, and saves results to CSV.

Failure isolation strategy:
  - A single sentence generation failure does not skip the word — the
    remaining 4 attempts still run.
  - A complete word failure (all 5 attempts fail) is logged and skipped —
    the rest of the words still run.
  - All errors are recorded in the CSV row so nothing is silently lost.
  - Progress is checkpointed after every word so a crash mid-run can be
    resumed from the last saved row.
"""

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
from tqdm import tqdm

from dgp.config import ModelConfig
from dgp.providers import ModelProvider, GroqProvider, VLLMProvider
from dgp.tasks.translation import TranslationPipeline
from dgp.tasks.sentencegeneration import SentenceGenerationPipeline

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_PATH = Path("training_data.csv")
SENTENCES_PER_WORD = 5
MAX_WORKERS = 2  # number of parallel threads — tune based on API rate limits
LOG_PATH = Path("generation_errors.log")

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CSV schema helpers
# ============================================================================

CSV_COLUMNS = [
    "index",
    "word",
    "english_sentence",
    "kinyarwanda_word_translation",
    "kinyarwanda_sentence_translation",
    "substituted_word",
    "substituted_sentence",
    "kinyarwanda_word_score",
    "kinyarwanda_sentence_score",
    "full_output",
    "error",
]


def result_to_row(idx: int, word: str, result: dict) -> dict:
    """Map a pipeline result dict to a flat CSV row."""
    scores = result.get("scores", {})
    word_score = scores.get("translated_descriptive_word") or {}
    sentence_score = scores.get("translated_substituted_sentence") or {}

    return {
        "index": idx,
        "word": word,
        "english_sentence": result.get("original_sentence", ""),
        "kinyarwanda_word_translation": result.get("translated_descriptive_word", ""),
        "kinyarwanda_sentence_translation": result.get("translated_substituted_sentence", ""),
        "substituted_word": result.get("descriptive_word", ""),
        "substituted_sentence": result.get("substituted_sentence", ""),
        "kinyarwanda_word_score": word_score.get("score", ""),
        "kinyarwanda_sentence_score": sentence_score.get("score", ""),
        "full_output": json.dumps(result, ensure_ascii=False),
        "error": "",
    }


def error_row(idx: int, word: str, error: str) -> dict:
    """Build a placeholder CSV row for a failed generation."""
    return {
        "index": idx,
        "word": word,
        "english_sentence": "",
        "kinyarwanda_word_translation": "",
        "kinyarwanda_sentence_translation": "",
        "substituted_word": "",
        "substituted_sentence": "",
        "kinyarwanda_word_score": "",
        "kinyarwanda_sentence_score": "",
        "full_output": "",
        "error": error,
    }


# ============================================================================
# Resume logic
# ============================================================================

def get_completed_indices(output_path: Path) -> set:
    """Return the set of (word, sentence_num) pairs already saved to CSV."""
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["index"])
        return set(existing["index"].tolist())
    except Exception:
        return set()


# ============================================================================
# Pipeline setup
# ============================================================================

def build_pipeline(
    provider: ModelProvider,
    generation_model_config: ModelConfig,
    translation_model_config: ModelConfig,
) -> SentenceGenerationPipeline:
    """
    Build the SentenceGenerationPipeline with explicit provider and model configs.

    Args:
        provider:                 The ModelProvider backend to use for all steps.
        generation_model_config:  ModelConfig for generation nodes (descriptive word,
                                  sentence generation, substitution, scoring).
        translation_model_config: ModelConfig for TranslationPipeline nodes
                                  (word and sentence translation).
    """
    translation_pipeline = TranslationPipeline(
        provider=provider,
        model_config=translation_model_config,
    )

    return SentenceGenerationPipeline(
        provider=provider,
        translation_pipeline=translation_pipeline,
        model_config=generation_model_config,
    )


# ============================================================================
# Main batch loop
# ============================================================================

def _process_word(
    word: str,
    base_idx: int,
    n_sentences: int,
    pipeline: SentenceGenerationPipeline,
    completed_indices: set,
    csv_lock: Lock,
    write_header_flag: list,
) -> int:
    """
    Process all sentence attempts for a single word.
    Runs in a worker thread. Uses csv_lock to safely write each row.

    Args:
        word:               The word to process.
        base_idx:           The starting global index for this word's sentences.
        n_sentences:        Number of sentence attempts to generate.
        pipeline:           The SentenceGenerationPipeline instance.
        completed_indices:  Set of already-saved row indices (for resume).
        csv_lock:           Threading lock that serialises all CSV writes.
        write_header_flag:  Single-element list used as a mutable bool to track
                            whether the CSV header has been written yet.

    Returns:
        Number of failed attempts for this word.
    """
    word_failures = 0

    for attempt in range(n_sentences):
        row_idx = base_idx + attempt

        if row_idx in completed_indices:
            continue

        try:
            result = pipeline.run(word=word)
            row = result_to_row(row_idx, word, result)

        except Exception as e:
            word_failures += 1
            error_msg = f"word='{word}' attempt={attempt} | {type(e).__name__}: {str(e)}"
            logger.error(error_msg + "\n" + traceback.format_exc())
            row = error_row(row_idx, word, error_msg)

        # Lock and flush — only one thread writes at a time
        with csv_lock:
            pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(
                OUTPUT_PATH,
                mode="a",
                header=write_header_flag[0],
                index=False,
                encoding="utf-8-sig",
            )
            write_header_flag[0] = False  # header written, never write again

    return word_failures


def run_batch(
    common_df: pd.DataFrame,
    pipeline: SentenceGenerationPipeline,
    n_sentences: int = SENTENCES_PER_WORD,
    max_workers: int = MAX_WORKERS,
):
    completed_indices = get_completed_indices(OUTPUT_PATH)

    # Mutable flag shared across threads — list wrapper makes it assignable
    write_header_flag = [not OUTPUT_PATH.exists()]

    # Lock that serialises all CSV writes across threads
    csv_lock = Lock()

    words = common_df["word"].tolist()

    # Pre-compute base index for each word (each word owns n_sentences indices)
    word_base_indices = {word: i * n_sentences for i, word in enumerate(words)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_word,
                word,
                word_base_indices[word],
                n_sentences,
                pipeline,
                completed_indices,
                csv_lock,
                write_header_flag,
            ): word
            for word in words
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Words"):
            word = futures[future]
            try:
                word_failures = future.result()
                if word_failures == n_sentences:
                    logger.error(
                        f"All {n_sentences} attempts failed for word='{word}' — skipping word entirely."
                    )
            except Exception as e:
                # Unexpected error escaping _process_word
                logger.error(f"Unhandled error for word='{word}': {type(e).__name__}: {str(e)}")

    print(f"\nDone. Results saved to: {OUTPUT_PATH}")
    print(f"Error log: {LOG_PATH}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # --- Build common_df (paste your existing notebook cells here) ----------
    from wordfreq import get_frequency_list, zipf_frequency
    import spacy

    nlp = spacy.load("en_core_web_sm")
    stopwords = nlp.Defaults.stop_words

    words_list = []
    commonality = []

    for words in get_frequency_list("en"):
        if len(words) < 1:
            continue
        for word in words:
            if word in stopwords:
                continue
            if not word.isalpha():
                continue
            words_list.append(word)
            commonality.append(zipf_frequency(word, "en"))

    common_df = pd.DataFrame({
        "word": words_list,
        "commonality": commonality
    }).sort_values("commonality")

    common_df = common_df.drop_duplicates(subset=["word"]).dropna()
    common_df["is_alpha"] = common_df["word"].str.isalpha()
    common_df = common_df[common_df["is_alpha"] == True]

    common_df = common_df[(common_df["commonality"]>3) & (common_df["commonality"]<4)]
    # common_df = common_df.head(2)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Provider and model configuration
    # -------------------------------------------------------------------------

    # Provider — swap out for OpenAIProvider, AnthropicProvider, etc. as needed
    # provider = GroqProvider()
    provider = VLLMProvider(
        base_url="http://localhost:10000/v1"
    )

    # Generation model: used for descriptive word, sentence generation,
    # substitution, and translation scoring nodes
    generation_model_config = ModelConfig(
        model_name="openai/gpt-oss-120b",
        temperature=0.7,
    )

    # Translation model: used for word and sentence translation nodes
    # Lower temperature for more deterministic translations
    translation_model_config = ModelConfig(
        model_name="openai/gpt-oss-120b",
        temperature=0.0,
    )

    # -------------------------------------------------------------------------
    # Build pipeline and run
    # -------------------------------------------------------------------------

    pipeline = build_pipeline(
        provider=provider,
        generation_model_config=generation_model_config,
        translation_model_config=translation_model_config,
    )

    run_batch(common_df, pipeline=pipeline)