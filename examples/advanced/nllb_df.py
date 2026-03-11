import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import threading
import logging
from tqdm import tqdm

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


def make_config(model_name: str, max_new_tokens: int) -> ModelConfig:
    return ModelConfig(
        model_name=model_name,
        temperature=0.0,
        max_tokens=max_new_tokens,
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
    Uses a thread-local store to give each worker thread its own pair of
    NLLBProvider instances. This avoids the 'Already borrowed' error that
    occurs when multiple threads share the same HuggingFace tokenizer,
    whose underlying Rust implementation is not thread-safe.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: int = -1,
        max_new_tokens: int = 512,
    ):
        self._model_name = model_name
        self._device = device
        self._config = make_config(model_name, max_new_tokens)
        self._local = threading.local()  # each thread gets its own .providers dict

    def _get_providers(self) -> dict:
        """Return this thread's provider pair, initialising them on first access."""
        if not hasattr(self._local, "providers"):
            logger.info("Thread %s: initialising providers…", threading.current_thread().name)
            self._local.providers = {
                "english": NLLBProvider(
                    model_name=self._model_name,
                    src_lang=NLLB_CODES["english"],
                    tgt_lang=NLLB_CODES["kinyarwanda"],
                    device=self._device,
                ),
                "kinyarwanda": NLLBProvider(
                    model_name=self._model_name,
                    src_lang=NLLB_CODES["kinyarwanda"],
                    tgt_lang=NLLB_CODES["english"],
                    device=self._device,
                ),
            }
        return self._local.providers

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
            provider = self._get_providers()[source_lang]
            translation = provider.invoke(text, system="", config=self._config)
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
        results: list[Optional[TranslationResult]] = [None] * len(df)
        logger.info("Translating %d rows with %d workers…", len(df), max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._translate_single, i, row[text_col], row[lang_col]): i
                for i, row in df.iterrows()
            }

            with tqdm(total=len(df), desc="Translating", unit="row") as pbar:
                for future in as_completed(future_to_idx):
                    result = future.result()
                    results[future_to_idx[future]] = result
                    pbar.set_postfix(lang=result.source_lang, status="ok" if not result.error else "err")
                    pbar.update(1)

        df = df.copy()
        df["translated_text"]   = [r.translation for r in results]
        df["target_language"]   = [r.target_lang  for r in results]
        df["translation_error"] = [r.error        for r in results]

        success = df["translation_error"].isna().sum()
        logger.info("Done — %d/%d succeeded.", success, len(df))
        return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a CSV/Parquet using NLLB.")
    parser.add_argument("input",  help="Path to input file (.csv or .parquet)")
    parser.add_argument("output", help="Path to write translated file (.csv or .parquet)")
    parser.add_argument("--text-col",       default="text",     help="Column with source text (default: text)")
    parser.add_argument("--lang-col",       default="language", help="Column with source language (default: language)")
    parser.add_argument("--max-workers",    type=int, default=4,   help="Thread pool size (default: 4)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row (default: 512)")
    parser.add_argument("--device",         type=int, default=-1,  help="-1 for CPU, >=0 for GPU index (default: -1)")
    parser.add_argument("--model",          default="facebook/nllb-200-distilled-600M", help="NLLB model name")
    args = parser.parse_args()

    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    logger.info("Loaded %d rows from %s", len(df), args.input)

    pipeline = TranslationPipeline(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    result_df = pipeline.run(
        df,
        max_workers=args.max_workers,
        text_col=args.text_col,
        lang_col=args.lang_col,
    )

    if args.output.endswith(".parquet"):
        result_df.to_parquet(args.output, index=False)
    else:
        result_df.to_csv(args.output, index=False)

    logger.info("Saved to %s", args.output)