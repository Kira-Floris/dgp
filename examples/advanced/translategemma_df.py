import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import threading
import logging
from tqdm import tqdm
import torch

from dgp.providers import TranslateGemmaProvider, ModelConfig
from dgp.tasks.translation import TranslationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ISO_CODES = {
    "english":     "en",
    "kinyarwanda": "rw",
}

TRANSLATION_TARGET = {
    "english":     "kinyarwanda",
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


def _build_pipeline(model_name: str, src_lang: str, tgt_lang: str, device: str, max_new_tokens: int) -> TranslationPipeline:
    """Instantiate a TranslateGemmaProvider and wrap it in a TranslationPipeline."""
    provider = TranslateGemmaProvider(
        model_name=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,
    )
    logger.info("Loaded %s → %s on device=%s", src_lang, tgt_lang, device)
    return TranslationPipeline(
        provider=provider,
        model_config=make_config(model_name, max_new_tokens),
    )


class ConcurrentTranslationPipeline:
    """
    Wraps two directional TranslationPipeline instances (EN→RW and RW→EN)
    and fans rows out to a thread pool. A per-direction threading.Lock ensures
    only one thread calls each pipeline at a time — HuggingFace's Rust
    tokenizer is not thread-safe.
    """

    def __init__(
        self,
        model_name: str = "google/translategemma-4b-it",
        device: str = "cpu",
        device_map: str | dict | None = None,
        max_new_tokens: int = 512,
    ):
        # device_map takes precedence — pass it through to the provider
        effective_device = device if device_map is None else None

        logger.info("Loading EN → RW pipeline…")
        en_pipeline = _build_pipeline(
            model_name, ISO_CODES["english"], ISO_CODES["kinyarwanda"],
            effective_device or device_map, max_new_tokens,
        )

        logger.info("Loading RW → EN pipeline…")
        rw_pipeline = _build_pipeline(
            model_name, ISO_CODES["kinyarwanda"], ISO_CODES["english"],
            effective_device or device_map, max_new_tokens,
        )

        self._guarded: dict[str, tuple[TranslationPipeline, threading.Lock]] = {
            "english":     (en_pipeline, threading.Lock()),
            "kinyarwanda": (rw_pipeline, threading.Lock()),
        }
        logger.info("Both pipelines ready.")

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
            lang_pipeline, lock = self._guarded[source_lang]
            with lock:
                result = lang_pipeline.run(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            return TranslationResult(
                index=index,
                source_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                translation=result["translation"],
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
    parser = argparse.ArgumentParser(description="Translate a CSV/Parquet using TranslateGemma.")
    parser.add_argument("input",  help="Path to input file (.csv or .parquet)")
    parser.add_argument("output", help="Path to write translated file (.csv or .parquet)")
    parser.add_argument("--text-col",       default="text",     help="Column with source text (default: text)")
    parser.add_argument("--lang-col",       default="language", help="Column with source language (default: language)")
    parser.add_argument("--max-workers",    type=int, default=4,   help="Thread pool size (default: 4)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row (default: 512)")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu",
                                            help="Single device: cpu, cuda, cuda:0, etc.")
    parser.add_argument("--device-map",     default=None,
                                            help="'auto' to shard across all GPUs (overrides --device)")
    parser.add_argument("--model",          default="google/translategemma-4b-it", help="TranslateGemma model name")
    args = parser.parse_args()

    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    logger.info("Loaded %d rows from %s", len(df), args.input)

    pipeline = ConcurrentTranslationPipeline(
        model_name=args.model,
        device=args.device,
        device_map=args.device_map,
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