import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import logging

from dgp.providers import NLLBProvider
from dgp.providers import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# NLLB language codes
LANG_TO_NLLB = {
    "english": "eng_Latn",
    "kinyarwanda": "kin_Latn",
}

# Translation direction: maps source language -> target language
TRANSLATION_TARGETS = {
    "english": "kinyarwanda",
    "kinyarwanda": "english",
}


@dataclass
class TranslationResult:
    index: int
    source_text: str
    source_lang: str
    target_lang: str
    translation: Optional[str] = None
    error: Optional[str] = None


def build_model_config(source_lang: str, target_lang: str) -> ModelConfig:
    """Build ModelConfig with NLLB forced BOS token for the target language."""
    return ModelConfig(
        model_name="facebook/nllb-200-distilled-600M",
        temperature=0.0,
        max_tokens=512,
        # Pass target language code so the provider sets forced_bos_token_id
        extra_params={"target_lang": LANG_TO_NLLB[target_lang]},
    )


def translate_single(
    provider: NLLBProvider,
    index: int,
    text: str,
    source_lang: str,
) -> TranslationResult:
    """Translate a single row. Designed to be called inside a thread."""
    source_lang = source_lang.strip().lower()
    target_lang = TRANSLATION_TARGETS.get(source_lang)

    if target_lang is None:
        return TranslationResult(
            index=index,
            source_text=text,
            source_lang=source_lang,
            target_lang="unknown",
            error=f"Unsupported source language: '{source_lang}'",
        )

    try:
        config = build_model_config(source_lang, target_lang)
        translation = provider.invoke(
            text,
            system="",
            config=config,
        )
        return TranslationResult(
            index=index,
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            translation=translation,
        )
    except Exception as exc:
        logger.warning("Row %d failed: %s", index, exc)
        return TranslationResult(
            index=index,
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            error=str(exc),
        )


def translate_dataframe(
    df: pd.DataFrame,
    max_workers: int = 4,
    text_col: str = "text",
    lang_col: str = "language",
) -> pd.DataFrame:
    """
    Translate every row in `df` concurrently.

    Parameters
    ----------
    df          : DataFrame with at least `text_col` and `lang_col` columns.
    max_workers : Number of threads to use.
    text_col    : Name of the column containing source text.
    lang_col    : Name of the column containing source language.

    Returns
    -------
    DataFrame with added columns: translated_text, target_language, translation_error.
    """
    provider = NLLBProvider()  # shared — NLLBProvider must be thread-safe for inference
    results: list[TranslationResult] = [None] * len(df)

    logger.info("Starting translation of %d rows with %d workers…", len(df), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                translate_single,
                provider,
                i,
                row[text_col],
                row[lang_col],
            ): i
            for i, row in df.iterrows()
        }

        completed = 0
        for future in as_completed(future_to_idx):
            result: TranslationResult = future.result()
            results[future_to_idx[future]] = result
            completed += 1
            if completed % 10 == 0 or completed == len(df):
                logger.info("Progress: %d/%d rows done", completed, len(df))

    # Reconstruct ordered results into new columns
    df = df.copy()
    df["translated_text"] = [r.translation for r in results]
    df["target_language"] = [r.target_lang for r in results]
    df["translation_error"] = [r.error for r in results]

    success = df["translation_error"].isna().sum()
    logger.info("Done. %d/%d translations succeeded.", success, len(df))
    return df


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "text": [
                "How are you doing?",
                "The weather is nice today.",
                "Muraho, amakuru yawe?",          # Kinyarwanda: "Hello, how are you?"
                "Ndashimye kubonana nawe.",        # Kinyarwanda: "I'm glad to meet you."
                "Machine learning is fascinating.",
            ],
            "language": [
                "english",
                "english",
                "kinyarwanda",
                "kinyarwanda",
                "english",
            ],
        }
    )

    print("Input DataFrame:")
    print(sample_data.to_string(index=False))
    print()

    result_df = translate_dataframe(sample_data, max_workers=4)

    print("\nOutput DataFrame:")
    print(result_df[["text", "language", "translated_text", "target_language", "translation_error"]].to_string(index=False))