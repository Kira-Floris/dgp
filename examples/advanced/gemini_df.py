# """
# translate_gemini.py
# -------------------
# Translates a bilingual (English ↔ Kinyarwanda) dataset using the
# TranslationPipeline + GeminiProvider from dgp.

# Mirrors the structure of translate_ollama.py:
#   - Splits dataset by language
#   - Sorts shortest-first for throughput
#   - Translates concurrently via ThreadPoolExecutor
#   - Writes each completed row to disk immediately (incremental CSV)
#   - Resumes automatically from a partial output file on re-run
#   - Retries failed rows with exponential back-off
#   - Chunks texts longer than MAX_WORDS to avoid token-limit errors
# """

# import argparse
# import csv
# import os
# import threading
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Optional

# import pandas as pd
# from tqdm import tqdm

# from dgp.config import ModelConfig
# from dgp.tasks.translation import TranslationPipeline
# from dgp.providers import GeminiProvider

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------

# DEFAULT_MODEL   = "gemini-2.0-flash"   # swap to gemini-1.5-pro etc. as needed
# MAX_RETRIES     = 3
# RETRY_DELAY     = 5                    # seconds; multiplied by attempt number
# MAX_WORDS       = 500                  # chunk texts longer than this

# SYSTEM_TEMPLATE = (
#     "Translate the following text from {src_lang} to {tgt_lang}. "
#     "Return ONLY the translated text — no explanation, no preamble, "
#     "no reasoning, no quotation marks."
# )

# LANGUAGE_NAMES = {
#     "english":     "English",
#     "kinyarwanda": "Kinyarwanda",
# }

# TRANSLATION_PAIR = {
#     "english":     "kinyarwanda",
#     "kinyarwanda": "english",
# }

# # ---------------------------------------------------------------------------
# # I/O helpers
# # ---------------------------------------------------------------------------

# def load_dataset(path: str) -> pd.DataFrame:
#     p = Path(path)
#     if p.is_dir():
#         from datasets import load_from_disk
#         return load_from_disk(str(p)).to_pandas()
#     ext = p.suffix.lower()
#     loaders = {
#         ".csv":     lambda: pd.read_csv(p),
#         ".parquet": lambda: pd.read_parquet(p),
#         ".json":    lambda: pd.read_json(p, lines=True),
#     }
#     if ext not in loaders:
#         raise ValueError(f"Unsupported extension {ext!r}. Use .csv, .parquet, or .json")
#     return loaders[ext]()


# def load_completed_indices(output_path: str) -> set[int]:
#     """Return set of _original_index values already saved with a non-empty translation."""
#     p = Path(output_path)
#     if not p.exists():
#         return set()
#     try:
#         done_df = pd.read_csv(p)
#         if "_original_index" in done_df.columns:
#             done_df = done_df[
#                 done_df["translation_text"].notna() &
#                 (done_df["translation_text"] != "")
#             ]
#             return set(done_df["_original_index"].tolist())
#     except Exception:
#         pass
#     return set()


# # ---------------------------------------------------------------------------
# # Incremental writer  (identical to translate_ollama.py)
# # ---------------------------------------------------------------------------

# class IncrementalCSVWriter:
#     """Thread-safe, append-mode CSV writer — one row at a time."""

#     def __init__(self, output_path: str, columns: list[str]) -> None:
#         self.path         = Path(output_path)
#         self.columns      = columns
#         self.lock         = threading.Lock()
#         self._initialised = self.path.exists()
#         self.path.parent.mkdir(parents=True, exist_ok=True)

#     def write_row(self, row: dict) -> None:
#         with self.lock:
#             file_exists = self.path.exists() and self._initialised
#             with open(self.path, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=self.columns)
#                 if not file_exists:
#                     writer.writeheader()
#                     self._initialised = True
#                 writer.writerow({col: row.get(col, "") for col in self.columns})


# # ---------------------------------------------------------------------------
# # Text chunking
# # ---------------------------------------------------------------------------

# def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
#     """Split long text at sentence boundaries into chunks of ≤ max_words words."""
#     words = text.split()
#     if len(words) <= max_words:
#         return [text]

#     chunks, current = [], []
#     for word in words:
#         current.append(word)
#         if len(current) >= max_words and word.endswith((".", "!", "?")):
#             chunks.append(" ".join(current))
#             current = []
#     if current:
#         chunks.append(" ".join(current))
#     return chunks


# # ---------------------------------------------------------------------------
# # Core translation (single chunk, with retries)
# # ---------------------------------------------------------------------------

# def _translate_chunk(
#     idx: int,
#     text: str,
#     src_lang: str,
#     tgt_lang: str,
#     pipeline: TranslationPipeline,
# ) -> tuple[int, str]:
#     """
#     Translate one text chunk via the TranslationPipeline.
#     Retries up to MAX_RETRIES times with exponential back-off.
#     """
#     last_error = None

#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             result = pipeline.run(
#                 text=text,
#                 source_lang=src_lang,
#                 target_lang=tgt_lang,
#                 system_template=SYSTEM_TEMPLATE,
#             )
#             translation = result.get("translation", "").strip()

#             if not translation:
#                 raise ValueError(f"Empty translation returned for idx={idx}")

#             return idx, translation

#         except Exception as e:
#             last_error = e
#             if attempt < MAX_RETRIES:
#                 time.sleep(RETRY_DELAY * attempt)   # 5s, 10s, 15s

#     raise RuntimeError(
#         f"Row {idx} failed after {MAX_RETRIES} attempts. "
#         f"Last error: {last_error}"
#     )


# def translate_one(
#     idx: int,
#     text: str,
#     src_lang: str,
#     tgt_lang: str,
#     pipeline: TranslationPipeline,
# ) -> tuple[int, str]:
#     """
#     Translate a single DataFrame row.
#     Long texts are split into chunks, translated separately, then rejoined.
#     """
#     chunks = chunk_text(text)
#     if len(chunks) == 1:
#         return _translate_chunk(idx, text, src_lang, tgt_lang, pipeline)

#     translated_chunks = [
#         _translate_chunk(idx, chunk, src_lang, tgt_lang, pipeline)[1]
#         for chunk in chunks
#     ]
#     return idx, " ".join(translated_chunks)


# # ---------------------------------------------------------------------------
# # Step 1 — Split + sort by length
# # ---------------------------------------------------------------------------

# def split_by_language(
#     df: pd.DataFrame,
#     lang_col: str,
#     text_col: str,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     lang_lower = df[lang_col].str.lower()
#     en_df  = df[lang_lower == "english"].copy()
#     kin_df = df[lang_lower == "kinyarwanda"].copy()

#     unknown_mask = ~lang_lower.isin({"english", "kinyarwanda"})
#     if unknown_mask.any():
#         unknown = df.loc[unknown_mask, lang_col].unique().tolist()
#         print(f"[WARN] Dropping {unknown_mask.sum():,} rows with unknown language(s): {unknown}")

#     # Shortest texts first — keeps threads busy and ETA accurate
#     en_df  = en_df.assign(_wc=en_df[text_col].str.split().str.len()).sort_values("_wc").drop(columns="_wc")
#     kin_df = kin_df.assign(_wc=kin_df[text_col].str.split().str.len()).sort_values("_wc").drop(columns="_wc")

#     print(f"Split  →  English: {len(en_df):,} rows  |  Kinyarwanda: {len(kin_df):,} rows")
#     return en_df, kin_df


# # ---------------------------------------------------------------------------
# # Step 2 — Translate via ThreadPoolExecutor with incremental saving
# # ---------------------------------------------------------------------------

# def translate_language_df(
#     sub_df: pd.DataFrame,
#     src_language: str,
#     text_col: str,
#     pipeline: TranslationPipeline,
#     model_name: str,
#     concurrency: int,
#     writer: IncrementalCSVWriter,
#     completed_indices: set[int],
# ) -> pd.DataFrame:
#     tgt_language = TRANSLATION_PAIR[src_language.lower()]
#     src_name     = LANGUAGE_NAMES[src_language.lower()]
#     tgt_name     = LANGUAGE_NAMES[tgt_language]

#     sub_df = sub_df.copy()
#     sub_df["translation_text"] = ""
#     sub_df["model_name"]       = model_name

#     if sub_df.empty:
#         return sub_df

#     # ── Resume: skip already-saved rows ───────────────────────────────────
#     pending_df = sub_df[~sub_df.index.isin(completed_indices)]
#     skipped    = len(sub_df) - len(pending_df)
#     if skipped:
#         print(f"  Resuming — skipping {skipped:,} already-translated rows")
#     if pending_df.empty:
#         print(f"  All {len(sub_df):,} rows already translated, nothing to do.")
#         return sub_df

#     texts        = pending_df[text_col].tolist()
#     orig_indices = pending_df.index.tolist()
#     total        = len(texts)

#     print(f"\n[Gemini] [{src_language} → {tgt_language}]  {total:,} rows to translate  "
#           f"|  concurrency {concurrency}  |  model {model_name!r}")

#     results:  dict[int, str] = {}
#     failures: dict[int, str] = {}

#     with ThreadPoolExecutor(max_workers=concurrency) as executor:
#         future_to_orig_idx = {
#             executor.submit(
#                 translate_one,
#                 orig_idx, text, src_name, tgt_name, pipeline,
#             ): orig_idx
#             for orig_idx, text in zip(orig_indices, texts)
#         }

#         with tqdm(
#             as_completed(future_to_orig_idx),
#             total=total,
#             desc=f"  {src_language[:2].upper()}→{tgt_language[:2].upper()}",
#             unit="row",
#             dynamic_ncols=True,
#             colour="cyan",
#         ) as pbar:
#             for future in pbar:
#                 orig_idx = future_to_orig_idx[future]
#                 try:
#                     _, translation = future.result()
#                     results[orig_idx] = translation

#                     # Flush to disk immediately
#                     row_data = pending_df.loc[orig_idx].to_dict()
#                     row_data["translation_text"] = translation
#                     row_data["model_name"]        = model_name
#                     row_data["_original_index"]   = orig_idx
#                     writer.write_row(row_data)

#                 except Exception as e:
#                     err_msg = str(e)
#                     failures[orig_idx] = err_msg
#                     pbar.write(f"  [FAIL] row {orig_idx}: {err_msg}")

#     if failures:
#         print(f"\n  ⚠  {len(failures):,} row(s) failed after {MAX_RETRIES} retries:")
#         for orig_idx, err in failures.items():
#             preview = str(pending_df.loc[orig_idx, text_col])[:80]
#             print(f"     row {orig_idx:>6}: {err}  |  text: {preview!r}")
#         print("  Re-run the script to retry — failed rows will be picked up automatically.\n")

#     for orig_idx, translation in results.items():
#         sub_df.at[orig_idx, "translation_text"] = translation

#     return sub_df


# # ---------------------------------------------------------------------------
# # Step 3 — Recombine
# # ---------------------------------------------------------------------------

# def recombine(en_df: pd.DataFrame, kin_df: pd.DataFrame) -> pd.DataFrame:
#     return pd.concat([en_df, kin_df]).sort_index()


# # ---------------------------------------------------------------------------
# # Full pipeline
# # ---------------------------------------------------------------------------

# def run_pipeline(
#     df: pd.DataFrame,
#     text_col: str,
#     lang_col: str,
#     model_name: str,
#     max_tokens: int,
#     concurrency: int,
#     output_path: str,
#     api_key: Optional[str] = None,
# ) -> pd.DataFrame:

#     # ── Build one shared pipeline (provider + graph compiled once) ─────────
#     provider = GeminiProvider(api_key=api_key)
#     config   = ModelConfig(
#         model_name=model_name,
#         temperature=0.0,
#         max_tokens=max_tokens,
#     )
#     pipeline = TranslationPipeline(provider=provider, model_config=config)

#     print(f"Provider    : {provider.get_provider_name()}")
#     print(f"Model       : {model_name}\n")

#     # ── Split ──────────────────────────────────────────────────────────────
#     en_df, kin_df = split_by_language(df, lang_col, text_col)

#     # ── Set up incremental writer + resume detection ───────────────────────
#     output_columns    = list(df.columns) + ["translation_text", "model_name", "_original_index"]
#     writer            = IncrementalCSVWriter(output_path, output_columns)
#     completed_indices = load_completed_indices(output_path)
#     if completed_indices:
#         print(f"Found {len(completed_indices):,} already-translated rows — will resume.\n")

#     # ── Translate ──────────────────────────────────────────────────────────
#     translated_en  = translate_language_df(
#         en_df,  "english",     text_col,
#         pipeline, model_name, concurrency, writer, completed_indices,
#     )
#     translated_kin = translate_language_df(
#         kin_df, "kinyarwanda", text_col,
#         pipeline, model_name, concurrency, writer, completed_indices,
#     )

#     result = recombine(translated_en, translated_kin)
#     print(f"Recombined → {len(result):,} rows total")
#     return result


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Translate a dataset (English ↔ Kinyarwanda) using Google Gemini.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("input",  help="Path to input file (.csv, .parquet, .json) or HF dataset dir")
#     parser.add_argument("output", help="Path to save the translated dataset")
#     parser.add_argument("--text-col",    default="text",          help="Column with source text")
#     parser.add_argument("--lang-col",    default="language",      help="Column with source language")
#     parser.add_argument("--model",       default=DEFAULT_MODEL,   help="Gemini model name")
#     parser.add_argument("--max-tokens",  type=int, default=2048,  help="max_output_tokens for Gemini")
#     parser.add_argument("--concurrency", type=int, default=8,     help="Max parallel Gemini requests")
#     parser.add_argument("--api-key",     default=None,            help="Gemini API key (overrides GEMINI_API_KEY env var)")
#     parser.add_argument("--format",      default="csv",
#                         choices=["parquet", "csv", "json"],       help="Output format")
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     print(f"Model       : {args.model}")
#     print(f"Max tokens  : {args.max_tokens}")
#     print(f"Concurrency : {args.concurrency}")
#     print(f"Input       : {args.input}")
#     print(f"Output      : {args.output}  [incremental CSV]")
#     print()

#     df = load_dataset(args.input)
#     print(f"Loaded {len(df):,} rows\n")

#     run_pipeline(
#         df,
#         text_col=args.text_col,
#         lang_col=args.lang_col,
#         model_name=args.model,
#         max_tokens=args.max_tokens,
#         concurrency=args.concurrency,
#         output_path=args.output,
#         api_key=args.api_key,
#     )

#     print(f"\nDone. Results saved incrementally to: {args.output}")


# if __name__ == "__main__":
#     main()

"""
translate_gemini.py
-------------------
Translates a bilingual (English ↔ Kinyarwanda) dataset using the
TranslationPipeline + GeminiProvider from dgp.

Mirrors the structure of translate_ollama.py:
  - Splits dataset by language
  - Sorts shortest-first for throughput
  - Translates concurrently via ThreadPoolExecutor
  - Writes each completed row to disk immediately (incremental CSV)
  - Resumes automatically from a partial output file on re-run
  - Retries failed rows with exponential back-off
  - Chunks texts longer than MAX_WORDS to avoid token-limit errors
"""

import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from dgp.config import ModelConfig
from dgp.tasks.translation import TranslationPipeline
from dgp.providers import GeminiProvider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "gemini-2.0-flash"   # swap to gemini-1.5-pro etc. as needed
MAX_RETRIES     = 3
RETRY_DELAY     = 5                    # seconds; multiplied by attempt number
MAX_WORDS       = 500                  # chunk texts longer than this

SYSTEM_TEMPLATE = (
    "Translate the following text from {src_lang} to {tgt_lang}. "
    "Return ONLY the translated text — no explanation, no preamble, "
    "no reasoning, no quotation marks."
)

LANGUAGE_NAMES = {
    "english":     "English",
    "kinyarwanda": "Kinyarwanda",
}

TRANSLATION_PAIR = {
    "english":     "kinyarwanda",
    "kinyarwanda": "english",
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
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


def load_completed_indices(output_path: str) -> set[int]:
    """Return set of _original_index values already saved with a non-empty translation."""
    p = Path(output_path)
    if not p.exists():
        return set()
    try:
        done_df = pd.read_csv(p)
        if "_original_index" in done_df.columns:
            done_df = done_df[
                done_df["translation_text"].notna() &
                (done_df["translation_text"] != "")
            ]
            return set(done_df["_original_index"].tolist())
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Incremental writer  (identical to translate_ollama.py)
# ---------------------------------------------------------------------------

class IncrementalCSVWriter:
    """Thread-safe, append-mode CSV writer — one row at a time."""

    def __init__(self, output_path: str, columns: list[str]) -> None:
        self.path         = Path(output_path)
        self.columns      = columns
        self.lock         = threading.Lock()
        self._initialised = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_row(self, row: dict) -> None:
        with self.lock:
            file_exists = self.path.exists() and self._initialised
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                if not file_exists:
                    writer.writeheader()
                    self._initialised = True
                writer.writerow({col: row.get(col, "") for col in self.columns})


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split long text at sentence boundaries into chunks of ≤ max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= max_words and word.endswith((".", "!", "?")):
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Core translation (single chunk, with retries)
# ---------------------------------------------------------------------------

def _translate_chunk(
    idx: int,
    text: str,
    src_lang: str,
    tgt_lang: str,
    pipeline: TranslationPipeline,
) -> tuple[int, str]:
    """
    Translate one text chunk via the TranslationPipeline.
    Retries up to MAX_RETRIES times with exponential back-off.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = pipeline.run(
                text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                system_template=SYSTEM_TEMPLATE,
            )
            translation = result.get("translation", "").strip()

            if not translation:
                raise ValueError(f"Empty translation returned for idx={idx}")

            return idx, translation

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)   # 5s, 10s, 15s

    raise RuntimeError(
        f"Row {idx} failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def translate_one(
    idx: int,
    text: str,
    src_lang: str,
    tgt_lang: str,
    pipeline: TranslationPipeline,
) -> tuple[int, str]:
    """
    Translate a single DataFrame row.
    Long texts are split into chunks, translated separately, then rejoined.
    """
    chunks = chunk_text(text)
    if len(chunks) == 1:
        return _translate_chunk(idx, text, src_lang, tgt_lang, pipeline)

    translated_chunks = [
        _translate_chunk(idx, chunk, src_lang, tgt_lang, pipeline)[1]
        for chunk in chunks
    ]
    return idx, " ".join(translated_chunks)


# ---------------------------------------------------------------------------
# Step 1 — Split + sort by length
# ---------------------------------------------------------------------------

def split_by_language(
    df: pd.DataFrame,
    lang_col: str,
    text_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lang_lower = df[lang_col].str.lower()
    en_df  = df[lang_lower == "english"].copy()
    kin_df = df[lang_lower == "kinyarwanda"].copy()

    unknown_mask = ~lang_lower.isin({"english", "kinyarwanda"})
    if unknown_mask.any():
        unknown = df.loc[unknown_mask, lang_col].unique().tolist()
        print(f"[WARN] Dropping {unknown_mask.sum():,} rows with unknown language(s): {unknown}")

    # Shortest texts first — keeps threads busy and ETA accurate
    en_df  = en_df.assign(_wc=en_df[text_col].str.split().str.len()).sort_values("_wc").drop(columns="_wc")
    kin_df = kin_df.assign(_wc=kin_df[text_col].str.split().str.len()).sort_values("_wc").drop(columns="_wc")

    print(f"Split  →  English: {len(en_df):,} rows  |  Kinyarwanda: {len(kin_df):,} rows")
    return en_df, kin_df


# ---------------------------------------------------------------------------
# Step 2 — Translate via ThreadPoolExecutor with incremental saving
# ---------------------------------------------------------------------------

def translate_language_df(
    sub_df: pd.DataFrame,
    src_language: str,
    text_col: str,
    pipeline: TranslationPipeline,
    model_name: str,
    concurrency: int,
    writer: IncrementalCSVWriter,
    completed_indices: set[int],
) -> pd.DataFrame:
    tgt_language = TRANSLATION_PAIR[src_language.lower()]
    src_name     = LANGUAGE_NAMES[src_language.lower()]
    tgt_name     = LANGUAGE_NAMES[tgt_language]

    sub_df = sub_df.copy()
    sub_df["translation_text"] = ""
    sub_df["model_name"]       = model_name

    if sub_df.empty:
        return sub_df

    # ── Resume: skip already-saved rows ───────────────────────────────────
    pending_df = sub_df[~sub_df.index.isin(completed_indices)]
    skipped    = len(sub_df) - len(pending_df)
    if skipped:
        print(f"  Resuming — skipping {skipped:,} already-translated rows")
    if pending_df.empty:
        print(f"  All {len(sub_df):,} rows already translated, nothing to do.")
        return sub_df

    texts        = pending_df[text_col].tolist()
    orig_indices = pending_df.index.tolist()
    total        = len(texts)

    print(f"\n[Gemini] [{src_language} → {tgt_language}]  {total:,} rows to translate  "
          f"|  concurrency {concurrency}  |  model {model_name!r}")

    results:  dict[int, str] = {}
    failures: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_orig_idx = {
            executor.submit(
                translate_one,
                orig_idx, text, src_name, tgt_name, pipeline,
            ): orig_idx
            for orig_idx, text in zip(orig_indices, texts)
        }

        with tqdm(
            as_completed(future_to_orig_idx),
            total=total,
            desc=f"  {src_language[:2].upper()}→{tgt_language[:2].upper()}",
            unit="row",
            dynamic_ncols=True,
            colour="cyan",
        ) as pbar:
            for future in pbar:
                orig_idx = future_to_orig_idx[future]
                try:
                    _, translation = future.result()
                    results[orig_idx] = translation

                    # Flush to disk immediately
                    row_data = pending_df.loc[orig_idx].to_dict()
                    row_data["translation_text"] = translation
                    row_data["model_name"]        = model_name
                    row_data["_original_index"]   = orig_idx
                    writer.write_row(row_data)

                except Exception as e:
                    err_msg = str(e)
                    failures[orig_idx] = err_msg
                    pbar.write(f"  [FAIL] row {orig_idx}: {err_msg}")

    if failures:
        print(f"\n  ⚠  {len(failures):,} row(s) failed after {MAX_RETRIES} retries:")
        for orig_idx, err in failures.items():
            preview = str(pending_df.loc[orig_idx, text_col])[:80]
            print(f"     row {orig_idx:>6}: {err}  |  text: {preview!r}")
        print("  Re-run the script to retry — failed rows will be picked up automatically.\n")

    for orig_idx, translation in results.items():
        sub_df.at[orig_idx, "translation_text"] = translation

    return sub_df


# ---------------------------------------------------------------------------
# Step 3 — Recombine
# ---------------------------------------------------------------------------

def recombine(en_df: pd.DataFrame, kin_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([en_df, kin_df]).sort_index()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    text_col: str,
    lang_col: str,
    model_name: str,
    max_tokens: int,
    concurrency: int,
    output_path: str,
) -> pd.DataFrame:

    # ── Build one shared pipeline (provider reads GEMINI_API_KEY from env) ─
    provider = GeminiProvider()  # reads GEMINI_API_KEY automatically
    config   = ModelConfig(
        model_name=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    pipeline = TranslationPipeline(provider=provider, model_config=config)

    print(f"Provider    : {provider.get_provider_name()}")
    print(f"Model       : {model_name}\n")

    # ── Split ──────────────────────────────────────────────────────────────
    en_df, kin_df = split_by_language(df, lang_col, text_col)

    # ── Set up incremental writer + resume detection ───────────────────────
    output_columns    = list(df.columns) + ["translation_text", "model_name", "_original_index"]
    writer            = IncrementalCSVWriter(output_path, output_columns)
    completed_indices = load_completed_indices(output_path)
    if completed_indices:
        print(f"Found {len(completed_indices):,} already-translated rows — will resume.\n")

    # ── Translate ──────────────────────────────────────────────────────────
    translated_en  = translate_language_df(
        en_df,  "english",     text_col,
        pipeline, model_name, concurrency, writer, completed_indices,
    )
    translated_kin = translate_language_df(
        kin_df, "kinyarwanda", text_col,
        pipeline, model_name, concurrency, writer, completed_indices,
    )

    result = recombine(translated_en, translated_kin)
    print(f"Recombined → {len(result):,} rows total")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a dataset (English ↔ Kinyarwanda) using Google Gemini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Path to input file (.csv, .parquet, .json) or HF dataset dir")
    parser.add_argument("output", help="Path to save the translated dataset")
    parser.add_argument("--text-col",    default="text",          help="Column with source text")
    parser.add_argument("--lang-col",    default="language",      help="Column with source language")
    parser.add_argument("--model",       default=DEFAULT_MODEL,   help="Gemini model name")
    parser.add_argument("--max-tokens",  type=int, default=2048,  help="max_output_tokens for Gemini")
    parser.add_argument("--concurrency", type=int, default=8,     help="Max parallel Gemini requests")
    parser.add_argument("--format",      default="csv",
                        choices=["parquet", "csv", "json"],       help="Output format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Fail early with a clear message if the env var is missing
    import os
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Export it before running:  export GEMINI_API_KEY='your-key-here'"
        )

    print(f"Model       : {args.model}")
    print(f"Max tokens  : {args.max_tokens}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Input       : {args.input}")
    print(f"Output      : {args.output}  [incremental CSV]")
    print()

    df = load_dataset(args.input)
    print(f"Loaded {len(df):,} rows\n")

    run_pipeline(
        df,
        text_col=args.text_col,
        lang_col=args.lang_col,
        model_name=args.model,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        output_path=args.output,
    )

    print(f"\nDone. Results saved incrementally to: {args.output}")


if __name__ == "__main__":
    main()