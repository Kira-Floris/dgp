# # # import argparse
# # # import pandas as pd
# # # from concurrent.futures import ThreadPoolExecutor, as_completed
# # # from dataclasses import dataclass
# # # from typing import Optional
# # # import threading
# # # import logging
# # # from tqdm import tqdm
# # # import torch

# # # from dgp.providers import NLLBProvider, ModelConfig

# # # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# # # logger = logging.getLogger(__name__)

# # # NLLB_CODES = {
# # #     "english": "eng_Latn",
# # #     "kinyarwanda": "kin_Latn",
# # # }

# # # TRANSLATION_TARGET = {
# # #     "english": "kinyarwanda",
# # #     "kinyarwanda": "english",
# # # }


# # # def make_config(model_name: str, max_new_tokens: int) -> ModelConfig:
# # #     return ModelConfig(
# # #         model_name=model_name,
# # #         temperature=0.0,
# # #         max_tokens=max_new_tokens,
# # #     )


# # # @dataclass
# # # class TranslationResult:
# # #     index: int
# # #     source_text: str
# # #     source_lang: str
# # #     target_lang: str
# # #     translation: Optional[str] = None
# # #     error: Optional[str] = None


# # # def _build_provider(model_name: str, src_lang: str, tgt_lang: str, device: int) -> NLLBProvider:
# # #     """
# # #     Load a provider on the main thread and do a warm-up forward pass to
# # #     confirm the weights are fully materialised before any worker touches it.
# # #     Raises immediately if the model is on meta device.
# # #     """
# # #     provider = NLLBProvider(
# # #         model_name=model_name,
# # #         src_lang=src_lang,
# # #         tgt_lang=tgt_lang,
# # #         device=device,
# # #     )

# # #     # Verify weights are real — meta tensors have no storage
# # #     first_param = next(provider.translator.model.parameters())
# # #     if first_param.is_meta:
# # #         raise RuntimeError(
# # #             f"Model loaded onto meta device (no real weights). "
# # #             f"Check that '{model_name}' downloaded correctly and that "
# # #             f"device={device} is valid on this machine."
# # #         )

# # #     logger.info("Loaded %s → %s (device=%s, dtype=%s)", src_lang, tgt_lang, first_param.device, first_param.dtype)
# # #     return provider


# # # class TranslationPipeline:
# # #     """
# # #     Providers are built eagerly on the main thread so weight loading and
# # #     verification happen before any worker thread runs. Workers receive
# # #     already-loaded providers via a per-direction threading.Lock so that
# # #     only one thread uses each provider at a time — HuggingFace's Rust
# # #     tokenizer is not thread-safe (causes 'Already borrowed' / meta-tensor
# # #     errors when accessed concurrently).
# # #     """

# # #     def __init__(
# # #         self,
# # #         model_name: str = "facebook/nllb-200-distilled-600M",
# # #         device: int = -1,
# # #         max_new_tokens: int = 512,
# # #     ):
# # #         self._config = make_config(model_name, max_new_tokens)

# # #         logger.info("Loading EN → RW provider…")
# # #         en_provider = _build_provider(model_name, NLLB_CODES["english"],     NLLB_CODES["kinyarwanda"], device)

# # #         logger.info("Loading RW → EN provider…")
# # #         rw_provider = _build_provider(model_name, NLLB_CODES["kinyarwanda"], NLLB_CODES["english"],     device)

# # #         # Pair each provider with a lock so threads queue up instead of colliding
# # #         self._guarded: dict[str, tuple[NLLBProvider, threading.Lock]] = {
# # #             "english":     (en_provider, threading.Lock()),
# # #             "kinyarwanda": (rw_provider, threading.Lock()),
# # #         }
# # #         logger.info("Both providers ready.")

# # #     def _translate_single(self, index: int, text: str, source_lang: str) -> TranslationResult:
# # #         source_lang = source_lang.strip().lower()
# # #         target_lang = TRANSLATION_TARGET.get(source_lang)

# # #         if target_lang is None:
# # #             return TranslationResult(
# # #                 index=index,
# # #                 source_text=text,
# # #                 source_lang=source_lang,
# # #                 target_lang="unknown",
# # #                 error=f"Unsupported language: '{source_lang}'",
# # #             )

# # #         try:
# # #             provider, lock = self._guarded[source_lang]
# # #             with lock:
# # #                 translation = provider.invoke(text, system="", config=self._config)
# # #             return TranslationResult(
# # #                 index=index,
# # #                 source_text=text,
# # #                 source_lang=source_lang,
# # #                 target_lang=target_lang,
# # #                 translation=translation,
# # #             )
# # #         except Exception as exc:
# # #             logger.warning("Row %d failed: %s", index, exc)
# # #             return TranslationResult(
# # #                 index=index,
# # #                 source_text=text,
# # #                 source_lang=source_lang,
# # #                 target_lang=target_lang,
# # #                 error=str(exc),
# # #             )

# # #     def run(
# # #         self,
# # #         df: pd.DataFrame,
# # #         max_workers: int = 4,
# # #         text_col: str = "text",
# # #         lang_col: str = "language",
# # #     ) -> pd.DataFrame:
# # #         results: list[Optional[TranslationResult]] = [None] * len(df)
# # #         logger.info("Translating %d rows with %d workers…", len(df), max_workers)

# # #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
# # #             future_to_idx = {
# # #                 executor.submit(self._translate_single, i, row[text_col], row[lang_col]): i
# # #                 for i, row in df.iterrows()
# # #             }

# # #             with tqdm(total=len(df), desc="Translating", unit="row") as pbar:
# # #                 for future in as_completed(future_to_idx):
# # #                     result = future.result()
# # #                     results[future_to_idx[future]] = result
# # #                     pbar.set_postfix(lang=result.source_lang, status="ok" if not result.error else "err")
# # #                     pbar.update(1)

# # #         df = df.copy()
# # #         df["translated_text"]   = [r.translation for r in results]
# # #         df["target_language"]   = [r.target_lang  for r in results]
# # #         df["translation_error"] = [r.error        for r in results]

# # #         success = df["translation_error"].isna().sum()
# # #         logger.info("Done — %d/%d succeeded.", success, len(df))
# # #         return df


# # # # ---------------------------------------------------------------------------
# # # # CLI entrypoint
# # # # ---------------------------------------------------------------------------
# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description="Translate a CSV/Parquet using NLLB.")
# # #     parser.add_argument("input",  help="Path to input file (.csv or .parquet)")
# # #     parser.add_argument("output", help="Path to write translated file (.csv or .parquet)")
# # #     parser.add_argument("--text-col",       default="text",     help="Column with source text (default: text)")
# # #     parser.add_argument("--lang-col",       default="language", help="Column with source language (default: language)")
# # #     parser.add_argument("--max-workers",    type=int, default=4,   help="Thread pool size (default: 4)")
# # #     parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row (default: 512)")
# # #     parser.add_argument("--device",         type=int, default=-1,  help="-1 for CPU, >=0 for GPU index (default: -1)")
# # #     parser.add_argument("--model",          default="facebook/nllb-200-distilled-600M", help="NLLB model name")
# # #     args = parser.parse_args()

# # #     if args.input.endswith(".parquet"):
# # #         df = pd.read_parquet(args.input)
# # #     else:
# # #         df = pd.read_csv(args.input)

# # #     logger.info("Loaded %d rows from %s", len(df), args.input)

# # #     pipeline = TranslationPipeline(
# # #         model_name=args.model,
# # #         device=args.device,
# # #         max_new_tokens=args.max_new_tokens,
# # #     )
# # #     result_df = pipeline.run(
# # #         df,
# # #         max_workers=args.max_workers,
# # #         text_col=args.text_col,
# # #         lang_col=args.lang_col,
# # #     )

# # #     if args.output.endswith(".parquet"):
# # #         result_df.to_parquet(args.output, index=False)
# # #     else:
# # #         result_df.to_csv(args.output, index=False)

# # #     logger.info("Saved to %s", args.output)

# # import argparse
# # import pandas as pd
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # from dataclasses import dataclass
# # from typing import Optional
# # import threading
# # import logging
# # from tqdm import tqdm
# # import os

# # from dgp.providers import NLLBProvider, ModelConfig
# # from dgp.tasks.translation import TranslationPipeline

# # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# # logger = logging.getLogger(__name__)

# # NLLB_CODES = {
# #     "english":     "eng_Latn",
# #     "kinyarwanda": "kin_Latn",
# # }

# # TRANSLATION_TARGET = {
# #     "english":     "kinyarwanda",
# #     "kinyarwanda": "english",
# # }


# # def make_config(model_name: str, max_new_tokens: int) -> ModelConfig:
# #     return ModelConfig(
# #         model_name=model_name,
# #         temperature=0.0,
# #         max_tokens=max_new_tokens,
# #     )


# # @dataclass
# # class TranslationResult:
# #     index: int
# #     source_text: str
# #     source_lang: str
# #     target_lang: str
# #     translation: Optional[str] = None
# #     error: Optional[str] = None


# # def _build_pipeline(model_name: str, src_lang: str, tgt_lang: str, device: int, max_new_tokens: int) -> TranslationPipeline:
# #     """Instantiate an NLLBProvider, verify weights, then wrap in a TranslationPipeline."""
# #     provider = NLLBProvider(
# #         model_name=model_name,
# #         src_lang=src_lang,
# #         tgt_lang=tgt_lang,
# #         device=device,
# #     )

# #     first_param = next(provider.translator.model.parameters())
# #     if first_param.is_meta:
# #         raise RuntimeError(
# #             f"Model loaded onto meta device (no real weights). "
# #             f"Check that '{model_name}' downloaded correctly and that "
# #             f"device={device} is valid on this machine."
# #         )

# #     logger.info("Loaded %s → %s (device=%s, dtype=%s)", src_lang, tgt_lang, first_param.device, first_param.dtype)
# #     return TranslationPipeline(
# #         provider=provider,
# #         model_config=make_config(model_name, max_new_tokens),
# #     )


# # class ConcurrentTranslationPipeline:
# #     """
# #     Providers are built eagerly on the main thread so weight loading and
# #     verification happen before any worker thread runs. Workers receive
# #     already-loaded TranslationPipeline instances via a per-direction
# #     threading.Lock so that only one thread uses each pipeline at a time —
# #     HuggingFace's Rust tokenizer is not thread-safe.
# #     """

# #     def __init__(
# #         self,
# #         model_name: str = "facebook/nllb-200-distilled-600M",
# #         device: int = -1,
# #         max_new_tokens: int = 512,
# #     ):
# #         logger.info("Loading EN → RW pipeline…")
# #         en_pipeline = _build_pipeline(model_name, NLLB_CODES["english"],     NLLB_CODES["kinyarwanda"], device, max_new_tokens)
# #         self.model_name = model_name

# #         logger.info("Loading RW → EN pipeline…")
# #         rw_pipeline = _build_pipeline(model_name, NLLB_CODES["kinyarwanda"], NLLB_CODES["english"],     device, max_new_tokens)

# #         self._guarded: dict[str, tuple[TranslationPipeline, threading.Lock]] = {
# #             "english":     (en_pipeline, threading.Lock()),
# #             "kinyarwanda": (rw_pipeline, threading.Lock()),
# #         }
# #         logger.info("Both pipelines ready.")

# #     def _translate_single(self, index: int, text: str, source_lang: str) -> TranslationResult:
# #         source_lang = source_lang.strip().lower()
# #         target_lang = TRANSLATION_TARGET.get(source_lang)

# #         if target_lang is None:
# #             return TranslationResult(
# #                 index=index,
# #                 source_text=text,
# #                 source_lang=source_lang,
# #                 target_lang="unknown",
# #                 error=f"Unsupported language: '{source_lang}'",
# #             )

# #         try:
# #             lang_pipeline, lock = self._guarded[source_lang]
# #             with lock:
# #                 result = lang_pipeline.run(
# #                     text=text,
# #                     source_lang=source_lang,
# #                     target_lang=target_lang,
# #                 )
# #             return TranslationResult(
# #                 index=index,
# #                 source_text=text,
# #                 source_lang=source_lang,
# #                 target_lang=target_lang,
# #                 translation=result["translation"],
# #             )
# #         except Exception as exc:
# #             logger.warning("Row %d failed: %s", index, exc)
# #             return TranslationResult(
# #                 index=index,
# #                 source_text=text,
# #                 source_lang=source_lang,
# #                 target_lang=target_lang,
# #                 error=str(exc),
# #             )

# #     def run(
# #         self,
# #         df: pd.DataFrame,
# #         max_workers: int = 4,
# #         text_col: str = "text",
# #         lang_col: str = "language",
# #         checkpoint_path: str | None = None,
# #     ) -> pd.DataFrame:
# #         results: list[Optional[TranslationResult]] = [None] * len(df)
# #         logger.info("Translating %d rows with %d workers…", len(df), max_workers)

# #         # Track whether the checkpoint file header has been written yet
# #         checkpoint_header_written = checkpoint_path is not None and os.path.exists(checkpoint_path)

# #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
# #             future_to_idx = {
# #                 executor.submit(self._translate_single, i, row[text_col], row[lang_col]): i
# #                 for i, row in df.iterrows()
# #             }

# #             with tqdm(total=len(df), desc="Translating", unit="row") as pbar:
# #                 for future in as_completed(future_to_idx):
# #                     result = future.result()
# #                     idx = future_to_idx[future]
# #                     results[idx] = result
# #                     pbar.set_postfix(lang=result.source_lang, status="ok" if not result.error else "err")
# #                     pbar.update(1)

# #                     # Append this single row to the checkpoint CSV immediately
# #                     if checkpoint_path is not None:
# #                         row_df = pd.DataFrame([{
# #                             text_col:            result.source_text,
# #                             lang_col:            result.source_lang,
# #                             "translated_text":   result.translation,
# #                             "target_language":   result.target_lang,
# #                             "translation_error": result.error,
# #                             "model_name":        self.model_name,
# #                         }])
# #                         row_df.to_csv(
# #                             checkpoint_path,
# #                             mode="a",
# #                             header=not checkpoint_header_written,
# #                             index=False,
# #                         )
# #                         checkpoint_header_written = True

# #         df = df.copy()
# #         df["translated_text"]    = [r.translation for r in results]
# #         df["target_language"]    = [r.target_lang  for r in results]
# #         df["translation_error"]  = [r.error        for r in results]
# #         df["model_name"]         = self.model_name

# #         success = df["translation_error"].isna().sum()
# #         logger.info("Done — %d/%d succeeded.", success, len(df))
# #         return df


# # # ---------------------------------------------------------------------------
# # # CLI entrypoint
# # # ---------------------------------------------------------------------------
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Translate a CSV/Parquet using NLLB.")
# #     parser.add_argument("input",  help="Path to input file (.csv or .parquet)")
# #     parser.add_argument("output", help="Path to write translated file (.csv or .parquet)")
# #     parser.add_argument("--text-col",       default="text",     help="Column with source text (default: text)")
# #     parser.add_argument("--lang-col",       default="language", help="Column with source language (default: language)")
# #     parser.add_argument("--max-workers",    type=int, default=4,   help="Thread pool size (default: 4)")
# #     parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row (default: 512)")
# #     parser.add_argument("--device",         type=int, default=-1,  help="-1 for CPU, >=0 for GPU index (default: -1)")
# #     parser.add_argument("--model",          default="facebook/nllb-200-distilled-600M", help="NLLB model name")
# #     args = parser.parse_args()

# #     if args.input.endswith(".parquet"):
# #         df = pd.read_parquet(args.input)
# #     else:
# #         df = pd.read_csv(args.input)

# #     logger.info("Loaded %d rows from %s", len(df), args.input)

# #     pipeline = ConcurrentTranslationPipeline(
# #         model_name=args.model,
# #         device=args.device,
# #         max_new_tokens=args.max_new_tokens,
# #     )
# #     result_df = pipeline.run(
# #         df,
# #         max_workers=args.max_workers,
# #         text_col=args.text_col,
# #         lang_col=args.lang_col,
# #     )

# #     if args.output.endswith(".parquet"):
# #         result_df.to_parquet(args.output, index=False)
# #     else:
# #         result_df.to_csv(args.output, index=False)

# #     logger.info("Saved to %s", args.output)

# import argparse
# import os
# import threading
# import logging
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataclasses import dataclass
# from typing import Optional

# from tqdm import tqdm
# from datasets import Dataset, load_dataset

# from dgp.providers import NLLBProvider, ModelConfig
# from dgp.tasks.translation import TranslationPipeline

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# NLLB_CODES = {
#     "english":     "eng_Latn",
#     "kinyarwanda": "kin_Latn",
# }

# TRANSLATION_TARGET = {
#     "english":     "kinyarwanda",
#     "kinyarwanda": "english",
# }


# def make_config(model_name: str, max_new_tokens: int) -> ModelConfig:
#     return ModelConfig(
#         model_name=model_name,
#         temperature=0.0,
#         max_tokens=max_new_tokens,
#     )


# @dataclass
# class TranslationResult:
#     index: int
#     source_text: str
#     source_lang: str
#     target_lang: str
#     translation: Optional[str] = None
#     error: Optional[str] = None


# def _build_pipeline(
#     model_name: str,
#     src_lang: str,
#     tgt_lang: str,
#     device: int,
#     max_new_tokens: int,
# ) -> TranslationPipeline:
#     provider = NLLBProvider(
#         model_name=model_name,
#         src_lang=src_lang,
#         tgt_lang=tgt_lang,
#         device=device,
#     )

#     first_param = next(provider.translator.model.parameters())
#     if first_param.is_meta:
#         raise RuntimeError(
#             f"Model loaded onto meta device (no real weights). "
#             f"Check that '{model_name}' downloaded correctly and that "
#             f"device={device} is valid on this machine."
#         )

#     logger.info("Loaded %s → %s (device=%s, dtype=%s)", src_lang, tgt_lang, first_param.device, first_param.dtype)
#     return TranslationPipeline(
#         provider=provider,
#         model_config=make_config(model_name, max_new_tokens),
#     )


# class ConcurrentTranslationPipeline:

#     def __init__(
#         self,
#         model_name: str = "facebook/nllb-200-distilled-600M",
#         device: int = -1,
#         max_new_tokens: int = 512,
#     ):
#         self.model_name = model_name

#         logger.info("Loading EN → RW pipeline…")
#         en_pipeline = _build_pipeline(model_name, NLLB_CODES["english"],     NLLB_CODES["kinyarwanda"], device, max_new_tokens)

#         logger.info("Loading RW → EN pipeline…")
#         rw_pipeline = _build_pipeline(model_name, NLLB_CODES["kinyarwanda"], NLLB_CODES["english"],     device, max_new_tokens)

#         self._guarded: dict[str, tuple[TranslationPipeline, threading.Lock]] = {
#             "english":     (en_pipeline, threading.Lock()),
#             "kinyarwanda": (rw_pipeline, threading.Lock()),
#         }
#         logger.info("Both pipelines ready.")

#     def _translate_single(self, index: int, text: str, source_lang: str) -> TranslationResult:
#         source_lang = source_lang.strip().lower()
#         target_lang = TRANSLATION_TARGET.get(source_lang)

#         if target_lang is None:
#             return TranslationResult(
#                 index=index,
#                 source_text=text,
#                 source_lang=source_lang,
#                 target_lang="unknown",
#                 error=f"Unsupported language: '{source_lang}'",
#             )

#         try:
#             lang_pipeline, lock = self._guarded[source_lang]
#             with lock:
#                 result = lang_pipeline.run(
#                     text=text,
#                     source_lang=source_lang,
#                     target_lang=target_lang,
#                 )
#             return TranslationResult(
#                 index=index,
#                 source_text=text,
#                 source_lang=source_lang,
#                 target_lang=target_lang,
#                 translation=result["translation"],
#             )
#         except Exception as exc:
#             logger.warning("Row %d failed: %s", index, exc)
#             return TranslationResult(
#                 index=index,
#                 source_text=text,
#                 source_lang=source_lang,
#                 target_lang=target_lang,
#                 error=str(exc),
#             )

#     def run(
#         self,
#         ds: Dataset,
#         max_workers: int = 4,
#         text_col: str = "text",
#         lang_col: str = "language",
#         checkpoint_path: str | None = None,
#     ) -> Dataset:
#         results: list[Optional[TranslationResult]] = [None] * len(ds)
#         logger.info("Translating %d rows with %d workers…", len(ds), max_workers)

#         checkpoint_header_written = checkpoint_path is not None and os.path.exists(checkpoint_path)

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # ds[i] returns a dict — access columns by key
#             future_to_idx = {
#                 executor.submit(self._translate_single, i, ds[i][text_col], ds[i][lang_col]): i
#                 for i in range(len(ds))
#             }

#             with tqdm(total=len(ds), desc="Translating", unit="row") as pbar:
#                 for future in as_completed(future_to_idx):
#                     idx = future_to_idx[future]
#                     try:
#                         result = future.result()
#                     except Exception as exc:
#                         result = TranslationResult(
#                             index=idx,
#                             source_text=ds[idx][text_col],
#                             source_lang=ds[idx][lang_col],
#                             target_lang=TRANSLATION_TARGET.get(ds[idx][lang_col].strip().lower(), "unknown"),
#                             error=str(exc),
#                         )
#                         logger.warning("Row %d raised unexpectedly: %s", idx, exc)

#                     results[idx] = result
#                     pbar.set_postfix(lang=result.source_lang, status="ok" if not result.error else "err")
#                     pbar.update(1)

#                     if checkpoint_path is not None:
#                         import pandas as pd
#                         row_df = pd.DataFrame([{
#                             text_col:            result.source_text,
#                             lang_col:            result.source_lang,
#                             "translated_text":   result.translation,
#                             "target_language":   result.target_lang,
#                             "translation_error": result.error,
#                             "model_name":        self.model_name,
#                         }])
#                         row_df.to_csv(
#                             checkpoint_path,
#                             mode="a",
#                             header=not checkpoint_header_written,
#                             index=False,
#                         )
#                         checkpoint_header_written = True

#         # Add new columns to the dataset
#         ds = ds.add_column("translated_text",   [r.translation for r in results])
#         ds = ds.add_column("target_language",   [r.target_lang  for r in results])
#         ds = ds.add_column("translation_error", [r.error        for r in results])
#         ds = ds.add_column("model_name",        [self.model_name] * len(ds))

#         success = sum(1 for r in results if r.error is None)
#         logger.info("Done — %d/%d succeeded.", success, len(ds))
#         return ds


# # ---------------------------------------------------------------------------
# # CLI entrypoint
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Translate a HuggingFace Dataset using NLLB.")
#     parser.add_argument("input",  help="Path to dataset (CSV, Parquet, or HF dataset directory)")
#     parser.add_argument("output", help="Path to save the translated dataset")
#     parser.add_argument("--text-col",       default="text",     help="Column with source text (default: text)")
#     parser.add_argument("--lang-col",       default="language", help="Column with source language (default: language)")
#     parser.add_argument("--max-workers",    type=int, default=4,   help="Thread pool size (default: 4)")
#     parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row (default: 512)")
#     parser.add_argument("--device",         type=int, default=-1,  help="-1 for CPU, >=0 for GPU index (default: -1)")
#     parser.add_argument("--model",          default="facebook/nllb-200-distilled-600M", help="NLLB model name")
#     parser.add_argument("--checkpoint",     default=None, help="Path to incrementally save completed rows (CSV)")
#     parser.add_argument("--format",         default="parquet", choices=["parquet", "csv", "json"],
#                                             help="Output format for the saved dataset (default: parquet)")
#     args = parser.parse_args()

#     # Load into a HuggingFace Dataset
#     if args.input.endswith(".csv"):
#         ds = load_dataset("csv",     data_files=args.input, split="train")
#     elif args.input.endswith(".parquet"):
#         ds = load_dataset("parquet", data_files=args.input, split="train")
#     elif args.input.endswith(".json") or args.input.endswith(".jsonl"):
#         ds = load_dataset("json",    data_files=args.input, split="train")
#     else:
#         # Assume it's a saved HF dataset directory
#         ds = Dataset.load_from_disk(args.input)

#     logger.info("Loaded %d rows from %s", len(ds), args.input)

#     pipeline = ConcurrentTranslationPipeline(
#         model_name=args.model,
#         device=args.device,
#         max_new_tokens=args.max_new_tokens,
#     )
#     result_ds = pipeline.run(
#         ds,
#         max_workers=args.max_workers,
#         text_col=args.text_col,
#         lang_col=args.lang_col,
#         checkpoint_path=args.checkpoint,
#     )

#     # Save dataset to disk
#     if args.format == "parquet":
#         result_ds.to_parquet(args.output)
#     elif args.format == "csv":
#         result_ds.to_csv(args.output)
#     elif args.format == "json":
#         result_ds.to_json(args.output)

#     logger.info("Saved to %s", args.output)

# # import argparse
# # import logging

# # import torch
# # from datasets import Dataset, load_dataset
# # from transformers import pipeline as hf_pipeline
# # from tqdm import tqdm

# # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# # logger = logging.getLogger(__name__)

# # NLLB_CODES = {
# #     "english":     "eng_Latn",
# #     "kinyarwanda": "kin_Latn",
# # }

# # TRANSLATION_TARGET = {
# #     "english":     "kinyarwanda",
# #     "kinyarwanda": "english",
# # }


# # class ConcurrentTranslationPipeline:
# #     """
# #     GPU-optimised translation pipeline. Avoids Dataset.map() entirely —
# #     that library uses its own 'multiprocess' fork which ignores Python's
# #     set_start_method and always forks, breaking CUDA.

# #     Instead, batches are sliced manually from the Dataset and fed directly
# #     to the HuggingFace pipeline in the main process, keeping everything
# #     single-process while fully utilising the GPU via batched inference.
# #     """

# #     def __init__(
# #         self,
# #         model_name: str = "facebook/nllb-200-distilled-600M",
# #         device: int = 0,
# #         max_new_tokens: int = 512,
# #         batch_size: int = 32,
# #     ):
# #         self.model_name     = model_name
# #         self.max_new_tokens = max_new_tokens
# #         self.batch_size     = batch_size

# #         dtype = torch.float16 if device >= 0 else torch.float32

# #         logger.info("Loading EN → RW pipeline on device=%d …", device)
# #         self._en_pipe = hf_pipeline(
# #             "translation",
# #             model=model_name,
# #             src_lang=NLLB_CODES["english"],
# #             tgt_lang=NLLB_CODES["kinyarwanda"],
# #             device=device,
# #             dtype=dtype,
# #         )

# #         logger.info("Loading RW → EN pipeline on device=%d …", device)
# #         self._rw_pipe = hf_pipeline(
# #             "translation",
# #             model=model_name,
# #             src_lang=NLLB_CODES["kinyarwanda"],
# #             tgt_lang=NLLB_CODES["english"],
# #             device=device,
# #             dtype=dtype,
# #         )

# #         logger.info("Both pipelines ready.")

# #     def _translate_batch(
# #         self,
# #         texts: list[str],
# #         langs: list[str],
# #     ) -> tuple[list, list, list]:
# #         """
# #         Translate one batch. Splits by source language, runs each group
# #         through its pipeline in a single forward pass, reassembles in order.
# #         Returns (translated, target_langs, errors).
# #         """
# #         n            = len(texts)
# #         translated   = [None] * n
# #         target_langs = [None] * n
# #         errors       = [None] * n

# #         groups: dict[str, list[int]] = {"english": [], "kinyarwanda": []}
# #         for i, lang in enumerate(langs):
# #             if lang in groups:
# #                 groups[lang].append(i)
# #             else:
# #                 errors[i]       = f"Unsupported language: '{lang}'"
# #                 target_langs[i] = "unknown"

# #         for src_lang, indices in groups.items():
# #             if not indices:
# #                 continue
# #             pipe      = self._en_pipe if src_lang == "english" else self._rw_pipe
# #             tgt_lang  = TRANSLATION_TARGET[src_lang]
# #             src_texts = [texts[i] for i in indices]

# #             try:
# #                 results = pipe(
# #                     src_texts,
# #                     max_length=self.max_new_tokens,
# #                     truncation=True,
# #                     batch_size=len(src_texts),
# #                 )
# #                 for j, i in enumerate(indices):
# #                     translated[i]   = results[j]["translation_text"].strip()
# #                     target_langs[i] = tgt_lang
# #             except Exception as exc:
# #                 logger.warning("Batch failed for %s: %s", src_lang, exc)
# #                 for i in indices:
# #                     errors[i]       = str(exc)
# #                     target_langs[i] = tgt_lang

# #         return translated, target_langs, errors

# #     def run(
# #         self,
# #         ds: Dataset,
# #         text_col: str = "text",
# #         lang_col: str = "language",
# #     ) -> Dataset:
# #         n = len(ds)
# #         logger.info("Translating %d rows (batch_size=%d)…", n, self.batch_size)

# #         all_translated   = []
# #         all_target_langs = []
# #         all_errors       = []
# #         all_model_names  = []

# #         num_batches = (n + self.batch_size - 1) // self.batch_size

# #         for batch_idx in tqdm(range(num_batches), desc="Translating", unit="batch"):
# #             start = batch_idx * self.batch_size
# #             end   = min(start + self.batch_size, n)

# #             # Dataset slicing returns a dict of lists
# #             batch      = ds[start:end]
# #             texts      = batch[text_col]
# #             langs      = [l.strip().lower() for l in batch[lang_col]]

# #             translated, target_langs, errors = self._translate_batch(texts, langs)

# #             all_translated.extend(translated)
# #             all_target_langs.extend(target_langs)
# #             all_errors.extend(errors)
# #             all_model_names.extend([self.model_name] * (end - start))

# #         ds = ds.add_column("translated_text",   all_translated)
# #         ds = ds.add_column("target_language",   all_target_langs)
# #         ds = ds.add_column("translation_error", all_errors)
# #         ds = ds.add_column("model_name",        all_model_names)

# #         success = sum(1 for e in all_errors if e is None)
# #         logger.info("Done — %d/%d succeeded.", success, n)
# #         return ds


# # # ---------------------------------------------------------------------------
# # # CLI entrypoint
# # # ---------------------------------------------------------------------------
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Translate a HuggingFace Dataset using NLLB on GPU.")
# #     parser.add_argument("input",  help="Path to input file (.csv, .parquet, .json) or HF dataset directory")
# #     parser.add_argument("output", help="Path to save the translated dataset")
# #     parser.add_argument("--text-col",       default="text",     help="Column with source text (default: text)")
# #     parser.add_argument("--lang-col",       default="language", help="Column with source language (default: language)")
# #     parser.add_argument("--batch-size",     type=int, default=32,  help="Rows per GPU batch (default: 32)")
# #     parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row (default: 512)")
# #     parser.add_argument("--device",         type=int, default=0,   help="GPU index, -1 for CPU (default: 0)")
# #     parser.add_argument("--model",          default="facebook/nllb-200-distilled-600M", help="NLLB model name")
# #     parser.add_argument("--format",         default="parquet", choices=["parquet", "csv", "json"],
# #                                             help="Output format (default: parquet)")
# #     args = parser.parse_args()

# #     if args.input.endswith(".csv"):
# #         ds = load_dataset("csv",     data_files=args.input, split="train")
# #     elif args.input.endswith(".parquet"):
# #         ds = load_dataset("parquet", data_files=args.input, split="train")
# #     elif args.input.endswith(".json") or args.input.endswith(".jsonl"):
# #         ds = load_dataset("json",    data_files=args.input, split="train")
# #     else:
# #         ds = Dataset.load_from_disk(args.input)

# #     logger.info("Loaded %d rows from %s", len(ds), args.input)

# #     pipeline = ConcurrentTranslationPipeline(
# #         model_name=args.model,
# #         device=args.device,
# #         max_new_tokens=args.max_new_tokens,
# #         batch_size=args.batch_size,
# #     )
# #     result_ds = pipeline.run(
# #         ds,
# #         text_col=args.text_col,
# #         lang_col=args.lang_col,
# #     )

# #     if args.format == "parquet":
# #         result_ds.to_parquet(args.output)
# #     elif args.format == "csv":
# #         result_ds.to_csv(args.output, index=False)
# #     elif args.format == "json":
# #         result_ds.to_json(args.output)

# #     logger.info("Saved to %s", args.output)

"""
Translate a HuggingFace Dataset / CSV / Parquet / JSON using NLLB on GPU.

Pipeline:
  1. Load full dataframe
  2. Split into two sub-dataframes by source language
       • English      → translate to Kinyarwanda
       • Kinyarwanda  → translate to English
  3. Run batched GPU inference on each sub-dataframe independently
  4. Recombine (preserving original row order) and save

Usage:
    python translate_nllb.py input.parquet output.parquet
    python translate_nllb.py input.csv output.csv --format csv --batch-size 16
    python translate_nllb.py input.json output.parquet --device -1
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB language codes
LANG_CODE = {
    "english":     "eng_Latn",
    "kinyarwanda": "kin_Latn",
}

# Bidirectional swap: source language → target language
TRANSLATION_PAIR = {
    "english":     "kinyarwanda",
    "kinyarwanda": "english",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV, Parquet, JSON, or HF-dataset directory into a DataFrame."""
    p = Path(path)
    if p.is_dir():
        from datasets import load_from_disk
        return load_from_disk(str(p)).to_pandas()
    ext = p.suffix.lower()
    loaders = {
        ".csv":     lambda: pd.read_csv(p),
        ".parquet": lambda: pd.read_parquet(p),
        ".json":    lambda: pd.read_json(p, lines=True),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported extension {ext!r}. Use .csv, .parquet, or .json")
    return loaders[ext]()


def save_dataset(df: pd.DataFrame, path: str, fmt: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(p, index=False)
    elif fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "json":
        df.to_json(p, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")
    print(f"\nSaved {len(df):,} rows → {p}")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device):
    print(f"\nLoading model {model_name!r} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print("Model ready.\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Step 1 — Split
# ---------------------------------------------------------------------------

def split_by_language(
    df: pd.DataFrame,
    lang_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (english_df, kinyarwanda_df).
    Both sub-dataframes keep the original integer index so rows can be
    recombined in their original order after translation.
    Rows with unknown languages are dropped with a warning.
    """
    lang_lower = df[lang_col].str.lower()

    en_df  = df[lang_lower == "english"].copy()
    kin_df = df[lang_lower == "kinyarwanda"].copy()

    unknown_mask = ~lang_lower.isin(LANG_CODE.keys())
    if unknown_mask.any():
        unknown = df.loc[unknown_mask, lang_col].unique().tolist()
        print(f"[WARN] Dropping {unknown_mask.sum():,} rows with unknown language(s): {unknown}")

    print(f"Split  →  English: {len(en_df):,} rows  |  Kinyarwanda: {len(kin_df):,} rows")
    return en_df, kin_df


# ---------------------------------------------------------------------------
# Step 2 — Translate one language sub-dataframe
# ---------------------------------------------------------------------------

def translate_language_df(
    sub_df: pd.DataFrame,
    src_language: str,          # "english" | "kinyarwanda"
    text_col: str,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Translate every row in sub_df from src_language to its paired language.
    Adds 'translation_text' and 'model_name' columns.
    Original index is preserved for later recombination.
    """
    tgt_language  = TRANSLATION_PAIR[src_language.lower()]
    src_lang_code = LANG_CODE[src_language.lower()]
    tgt_lang_code = LANG_CODE[tgt_language]
    tgt_lang_id   = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    sub_df = sub_df.copy()
    sub_df["translation_text"] = ""
    sub_df["model_name"]       = model_name

    texts   = sub_df[text_col].tolist()
    indices = sub_df.index.tolist()
    total   = len(texts)

    print(f"[{src_language} → {tgt_language}]  {total:,} rows  |  batch size {batch_size}")

    # Set src_lang once for the entire sub-dataframe (all rows share the same source)
    tokenizer.src_lang = src_lang_code

    translations: list[str] = []

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_new_tokens,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=max_new_tokens,
                # num_beams=4,
            )

        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        done = min(i + batch_size, total)
        print(f"  {done:>{len(str(total))}}/{total}  ({done / total * 100:.1f}%)", end="\r")

    print()  # newline after in-place progress line

    for idx, translation in zip(indices, translations):
        sub_df.at[idx, "translation_text"] = translation

    return sub_df


# ---------------------------------------------------------------------------
# Step 3 — Recombine
# ---------------------------------------------------------------------------

def recombine(en_df: pd.DataFrame, kin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate the two translated sub-dataframes and restore the original
    row order using the preserved integer index.
    """
    return pd.concat([en_df, kin_df]).sort_index()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    text_col: str,
    lang_col: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    model_name: str,
) -> pd.DataFrame:
    # ── 1. Split ──────────────────────────────────────────────────────────
    en_df, kin_df = split_by_language(df, lang_col)

    # ── 2. Load model once — shared across both translation passes ────────
    tokenizer, model = load_model(model_name, device)

    # ── 3. Translate each language sub-dataframe in turn ──────────────────
    translated_en = translate_language_df(
        en_df,  "english",     text_col,
        tokenizer, model, device, batch_size, max_new_tokens, model_name,
    )
    translated_kin = translate_language_df(
        kin_df, "kinyarwanda", text_col,
        tokenizer, model, device, batch_size, max_new_tokens, model_name,
    )

    # ── 4. Recombine into original row order ──────────────────────────────
    result = recombine(translated_en, translated_kin)
    print(f"Recombined → {len(result):,} rows total")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a HuggingFace Dataset using NLLB on GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Path to input file (.csv, .parquet, .json) or HF dataset directory")
    parser.add_argument("output", help="Path to save the translated dataset")
    parser.add_argument("--text-col",       default="text",        help="Column with source text")
    parser.add_argument("--lang-col",       default="language",    help="Column with source language")
    parser.add_argument("--batch-size",     type=int, default=32,  help="Rows per GPU batch")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row")
    parser.add_argument("--device",         type=int, default=0,   help="GPU index, -1 for CPU")
    parser.add_argument("--model",          default=DEFAULT_MODEL, help="NLLB model name or local path")
    parser.add_argument("--format",         default="csv",     choices=["parquet", "csv", "json"],
                                            help="Output format")
    return parser.parse_args()


def resolve_device(device_idx: int) -> torch.device:
    if device_idx == -1:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("WARN: CUDA not available — falling back to CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if device_idx >= n:
        raise ValueError(f"GPU index {device_idx} out of range (found {n} GPU(s))")
    return torch.device(f"cuda:{device_idx}")


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    print(f"Device : {device}")
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}  [{args.format}]")

    df = load_dataset(args.input)
    print(f"Loaded {len(df):,} rows")

    result = run_pipeline(
        df,
        text_col=args.text_col,
        lang_col=args.lang_col,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        model_name=args.model,
    )

    # Save once — after both language passes are complete
    save_dataset(result, args.output, args.format)


if __name__ == "__main__":
    main()