import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import logging

from dgp.providers import NLLBProvider, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NLLB_CODES = {
    "english": "eng_Latn",
    "kinyarwanda": "kin_Latn",
}

TRANSLATION_TARGET = {
    "english": "kinyarwanda",
    "kinyarwanda": "english",
}

MODEL_CONFIG = ModelConfig(
    model_name="facebook/nllb-200-distilled-600M",
    temperature=0.0,
    max_tokens=512,
)


@dataclass
class TranslationResult:
    index: int
    source_text: str
    source_lang: str
    target_lang: str
    translation: Optional[str] = None
    error: Optional[str] = None


class TranslationPipeline:
    """
    Holds two directional NLLBProvider instances and routes each row
    to the correct one based on the source language.
    """

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: int = -1):
        logger.info("Loading providers…")
        self._providers = {
            "english": NLLBProvider(
                model_name=model_name,
                src_lang=NLLB_CODES["english"],
                tgt_lang=NLLB_CODES["kinyarwanda"],
                device=device,
            ),
            "kinyarwanda": NLLBProvider(
                model_name=model_name,
                src_lang=NLLB_CODES["kinyarwanda"],
                tgt_lang=NLLB_CODES["english"],
                device=device,
            ),
        }
        logger.info("Both providers ready.")

    def _translate_single(self, index: int, text: str, source_lang: str) -> TranslationResult:
        source_lang = source_lang.strip().lower()
        target_lang = TRANSLATION_TARGET.get(source_lang)

        if target_lang is None:
            return TranslationResult(
                index=index,
                source_text=text,
                source_lang=source_lang,
                target_lang="unknown",
                error=f"Unsupported language: '{source_lang}'",
            )

        try:
            provider = self._providers[source_lang]
            translation = provider.invoke(text, system="", config=MODEL_CONFIG)
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

    def run(
        self,
        df: pd.DataFrame,
        max_workers: int = 4,
        text_col: str = "text",
        lang_col: str = "language",
    ) -> pd.DataFrame:
        """
        Translate every row in `df` concurrently.

        Returns the original DataFrame with three added columns:
            - translated_text
            - target_language
            - translation_error  (None on success)
        """
        results: list[Optional[TranslationResult]] = [None] * len(df)
        logger.info("Translating %d rows with %d workers…", len(df), max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._translate_single, i, row[text_col], row[lang_col]): i
                for i, row in df.iterrows()
            }

            for completed_count, future in enumerate(as_completed(future_to_idx), 1):
                result = future.result()
                results[future_to_idx[future]] = result
                if completed_count % 10 == 0 or completed_count == len(df):
                    logger.info("Progress: %d/%d", completed_count, len(df))

        df = df.copy()
        df["translated_text"]   = [r.translation for r in results]
        df["target_language"]   = [r.target_lang  for r in results]
        df["translation_error"] = [r.error        for r in results]

        success = df["translation_error"].isna().sum()
        logger.info("Done — %d/%d succeeded.", success, len(df))
        return df


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        "text": [
            "How are you doing?",
            "The weather is nice today.",
            "Muraho, amakuru yawe?",
            "Ndashimye kubonana nawe.",
            "Machine learning is fascinating.",
        ],
        "language": [
            "english",
            "english",
            "kinyarwanda",
            "kinyarwanda",
            "english",
        ],
    })

    pipeline = TranslationPipeline(device=-1)
    result_df = pipeline.run(sample, max_workers=4)

    print(result_df[["text", "language", "translated_text", "target_language", "translation_error"]].to_string(index=False))