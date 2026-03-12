# import argparse
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataclasses import dataclass
# from typing import Optional
# import threading
# import logging
# from tqdm import tqdm
# import torch
# import os

# from dgp.providers import TranslateGemmaProvider, ModelConfig
# from dgp.tasks.translation import TranslationPipeline

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# ISO_CODES = {
#     "english":     "en",
#     "kinyarwanda": "rw",
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


# def _build_pipeline(model_name: str, src_lang: str, tgt_lang: str, device: str, max_new_tokens: int) -> TranslationPipeline:
#     """Instantiate a TranslateGemmaProvider and wrap it in a TranslationPipeline."""
#     provider = TranslateGemmaProvider(
#         model_name=model_name,
#         src_lang=src_lang,
#         tgt_lang=tgt_lang,
#         device=device,
#     )
#     logger.info("Loaded %s → %s on device=%s", src_lang, tgt_lang, device)
#     return TranslationPipeline(
#         provider=provider,
#         model_config=make_config(model_name, max_new_tokens),
#     )


# class ConcurrentTranslationPipeline:
#     """
#     Wraps two directional TranslationPipeline instances (EN→RW and RW→EN)
#     and fans rows out to a thread pool. A per-direction threading.Lock ensures
#     only one thread calls each pipeline at a time — HuggingFace's Rust
#     tokenizer is not thread-safe.
#     """

#     def __init__(
#         self,
#         model_name: str = "google/translategemma-4b-it",
#         device: str = "cpu",
#         device_map: str | dict | None = None,
#         max_new_tokens: int = 512,
#     ):
#         # device_map takes precedence — pass it through to the provider
#         effective_device = device if device_map is None else None

#         self.model_name = model_name

#         logger.info("Loading EN → RW pipeline…")
#         en_pipeline = _build_pipeline(
#             model_name, ISO_CODES["english"], ISO_CODES["kinyarwanda"],
#             effective_device or device_map, max_new_tokens,
#         )

#         logger.info("Loading RW → EN pipeline…")
#         rw_pipeline = _build_pipeline(
#             model_name, ISO_CODES["kinyarwanda"], ISO_CODES["english"],
#             effective_device or device_map, max_new_tokens,
#         )

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
#         df: pd.DataFrame,
#         max_workers: int = 4,
#         text_col: str = "text",
#         lang_col: str = "language",
#         checkpoint_path: str | None = None,
#     ) -> pd.DataFrame:
#         results: list[Optional[TranslationResult]] = [None] * len(df)
#         logger.info("Translating %d rows with %d workers…", len(df), max_workers)

#         # Track whether the checkpoint file header has been written yet
#         checkpoint_header_written = checkpoint_path is not None and os.path.exists(checkpoint_path)

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             future_to_idx = {
#                 executor.submit(self._translate_single, i, row[text_col], row[lang_col]): i
#                 for i, row in df.iterrows()
#             }

#             with tqdm(total=len(df), desc="Translating", unit="row") as pbar:
#                 for future in as_completed(future_to_idx):
#                     result = future.result()
#                     idx = future_to_idx[future]
#                     results[idx] = result
#                     pbar.set_postfix(lang=result.source_lang, status="ok" if not result.error else "err")
#                     pbar.update(1)

#                     # Append this single row to the checkpoint CSV immediately
#                     if checkpoint_path is not None:
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

#         df = df.copy()
#         df["translated_text"]    = [r.translation for r in results]
#         df["target_language"]    = [r.target_lang  for r in results]
#         df["translation_error"]  = [r.error        for r in results]
#         df["model_name"]         = self.model_name

#         success = df["translation_error"].isna().sum()
#         logger.info("Done — %d/%d succeeded.", success, len(df))
#         return df


# # ---------------------------------------------------------------------------
# # CLI entrypoint
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Translate a CSV/Parquet using TranslateGemma.")
#     parser.add_argument("input",  help="Path to input file (.csv or .parquet)")
#     parser.add_argument("output", help="Path to write translated file (.csv or .parquet)")
#     parser.add_argument("--text-col",       default="text",     help="Column with source text (default: text)")
#     parser.add_argument("--lang-col",       default="language", help="Column with source language (default: language)")
#     parser.add_argument("--max-workers",    type=int, default=4,   help="Thread pool size (default: 4)")
#     parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row (default: 512)")
#     parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu",
#                                             help="Single device: cpu, cuda, cuda:0, etc.")
#     parser.add_argument("--device-map",     default=None,
#                                             help="'auto' to shard across all GPUs (overrides --device)")
#     parser.add_argument("--model",          default="google/translategemma-4b-it", help="TranslateGemma model name")
#     args = parser.parse_args()

#     if args.input.endswith(".parquet"):
#         df = pd.read_parquet(args.input)
#     else:
#         df = pd.read_csv(args.input)

#     logger.info("Loaded %d rows from %s", len(df), args.input)

#     pipeline = ConcurrentTranslationPipeline(
#         model_name=args.model,
#         device=args.device,
#         device_map=args.device_map,
#         max_new_tokens=args.max_new_tokens,
#     )
#     result_df = pipeline.run(
#         df,
#         max_workers=args.max_workers,
#         text_col=args.text_col,
#         lang_col=args.lang_col,
#     )

#     if args.output.endswith(".parquet"):
#         result_df.to_parquet(args.output, index=False)
#     else:
#         result_df.to_csv(args.output, index=False)

#     logger.info("Saved to %s", args.output)

"""
Translate a HuggingFace Dataset / CSV / Parquet / JSON using either:
  • NLLB   (facebook/nllb-200-distilled-600M)  — seq2seq, AutoTokenizer
  • TranslateGemma (google/translategemma-4b-it) — decoder VLM, AutoProcessor

Pipeline:
  1. Load full dataframe
  2. Split into two sub-dataframes by source language
       • English      → translate to Kinyarwanda
       • Kinyarwanda  → translate to English
  3. Run batched GPU inference on each sub-dataframe independently
  4. Recombine (preserving original row order) and save

Usage:
    # NLLB (default)
    python translate_nllb.py input.parquet output.parquet

    # TranslateGemma
    python translate_nllb.py input.parquet output.parquet --model google/translategemma-4b-it

    # Other options
    python translate_nllb.py input.csv output.csv --format csv --batch-size 8 --device 0
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB language codes  (BCP-47 style used by NLLB)
NLLB_LANG_CODE = {
    "english":     "eng_Latn",
    "kinyarwanda": "kin_Latn",
}

# TranslateGemma language codes  (ISO 639-1)
GEMMA_LANG_CODE = {
    "english":     "en",
    "kinyarwanda": "rw",
}

# Bidirectional swap: source language → target language
TRANSLATION_PAIR = {
    "english":     "kinyarwanda",
    "kinyarwanda": "english",
}


def is_translategemma(model_name: str) -> bool:
    return "translategemma" in model_name.lower()


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
# Model loaders — one per backend
# ---------------------------------------------------------------------------

def load_nllb(model_name: str, device: torch.device):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"\nLoading NLLB model {model_name!r} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print("Model ready.\n")
    return tokenizer, model


def load_translategemma(model_name: str, device: torch.device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"\nLoading TranslateGemma model {model_name!r} on {device} …")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,          # TranslateGemma requires bfloat16
        device_map={"": device},             # pin to the requested GPU
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Model ready.\n")
    return processor, model


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
    """
    lang_lower = df[lang_col].str.lower()

    en_df  = df[lang_lower == "english"].copy()
    kin_df = df[lang_lower == "kinyarwanda"].copy()

    unknown_mask = ~lang_lower.isin({"english", "kinyarwanda"})
    if unknown_mask.any():
        unknown = df.loc[unknown_mask, lang_col].unique().tolist()
        print(f"[WARN] Dropping {unknown_mask.sum():,} rows with unknown language(s): {unknown}")

    print(f"Split  →  English: {len(en_df):,} rows  |  Kinyarwanda: {len(kin_df):,} rows")
    return en_df, kin_df


# ---------------------------------------------------------------------------
# Step 2A — NLLB batch translation
# ---------------------------------------------------------------------------

def translate_language_df_nllb(
    sub_df: pd.DataFrame,
    src_language: str,
    text_col: str,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    model_name: str,
) -> pd.DataFrame:
    tgt_language  = TRANSLATION_PAIR[src_language.lower()]
    src_lang_code = NLLB_LANG_CODE[src_language.lower()]
    tgt_lang_code = NLLB_LANG_CODE[tgt_language]
    tgt_lang_id   = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    sub_df = sub_df.copy()
    sub_df["translation_text"] = ""
    sub_df["model_name"]       = model_name

    texts   = sub_df[text_col].tolist()
    indices = sub_df.index.tolist()
    total   = len(texts)

    print(f"[NLLB] [{src_language} → {tgt_language}]  {total:,} rows  |  batch {batch_size}")

    # src_lang set once — all rows share the same source language
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
                do_sample=False,
            )

        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        done = min(i + batch_size, total)
        print(f"  {done:>{len(str(total))}}/{total}  ({done / total * 100:.1f}%)", end="\r")

    print()

    sub_df.loc[indices, "translation_text"] = translations
    return sub_df


# ---------------------------------------------------------------------------
# Step 2B — TranslateGemma batch translation
#
# TranslateGemma's chat template is per-message, so we build one message
# per row, then process in batches using left-padding for uniform input_ids,
# and slice off the prompt tokens from each output.
# ---------------------------------------------------------------------------

def translate_language_df_gemma(
    sub_df: pd.DataFrame,
    src_language: str,
    text_col: str,
    processor,
    model,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    model_name: str,
) -> pd.DataFrame:
    tgt_language      = TRANSLATION_PAIR[src_language.lower()]
    src_lang_code     = GEMMA_LANG_CODE[src_language.lower()]
    tgt_lang_code     = GEMMA_LANG_CODE[tgt_language]

    sub_df = sub_df.copy()
    sub_df["translation_text"] = ""
    sub_df["model_name"]       = model_name

    texts   = sub_df[text_col].tolist()
    indices = sub_df.index.tolist()
    total   = len(texts)

    print(f"[TranslateGemma] [{src_language} → {tgt_language}]  {total:,} rows  |  batch {batch_size}")

    # Build one chat message per row up-front
    def make_message(text: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type":             "text",
                        "source_lang_code": src_lang_code,
                        "target_lang_code": tgt_lang_code,
                        "text":             text,
                    }
                ],
            }
        ]

    translations: list[str] = []

    # Use left-padding so all sequences in a batch end at the same position,
    # which keeps the prompt-length offset consistent for output slicing.
    processor.tokenizer.padding_side = "left"

    for i in range(0, total, batch_size):
        batch_texts    = texts[i : i + batch_size]
        batch_messages = [make_message(t) for t in batch_texts]

        # Apply the chat template to each message individually, then batch
        encoded_list = [
            processor.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            for msg in batch_messages
        ]

        # Pad to the longest sequence in the batch
        input_lengths = [enc["input_ids"].shape[1] for enc in encoded_list]
        max_len       = max(input_lengths)

        pad_id = processor.tokenizer.pad_token_id or 0

        padded_ids      = torch.full((len(batch_texts), max_len), pad_id, dtype=torch.long)
        padded_mask     = torch.zeros((len(batch_texts), max_len), dtype=torch.long)

        for j, (enc, length) in enumerate(zip(encoded_list, input_lengths)):
            pad_amt = max_len - length
            padded_ids[j, pad_amt:]  = enc["input_ids"][0]
            padded_mask[j, pad_amt:] = enc["attention_mask"][0]

        inputs = {
            "input_ids":      padded_ids.to(device),
            "attention_mask": padded_mask.to(device),
        }

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Slice off the prompt tokens — each row has a different prompt length
        for j, (out_seq, prompt_len) in enumerate(zip(outputs, input_lengths)):
            new_tokens = out_seq[max_len:]        # max_len = padded prompt length
            decoded    = processor.decode(new_tokens, skip_special_tokens=True)
            translations.append(decoded)

        done = min(i + batch_size, total)
        print(f"  {done:>{len(str(total))}}/{total}  ({done / total * 100:.1f}%)", end="\r")

    print()

    sub_df.loc[indices, "translation_text"] = translations
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
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    model_name: str,
) -> pd.DataFrame:
    # ── 1. Split ──────────────────────────────────────────────────────────
    en_df, kin_df = split_by_language(df, lang_col)

    # ── 2. Load model once — shared for both language passes ──────────────
    use_gemma = is_translategemma(model_name)

    if use_gemma:
        processor_or_tokenizer, model = load_translategemma(model_name, device)
        translate_fn = translate_language_df_gemma
    else:
        processor_or_tokenizer, model = load_nllb(model_name, device)
        translate_fn = translate_language_df_nllb

    # ── 3. Translate each language sub-dataframe ──────────────────────────
    translated_en = translate_fn(
        en_df,  "english",     text_col,
        processor_or_tokenizer, model, device,
        batch_size, max_new_tokens, model_name,
    )
    translated_kin = translate_fn(
        kin_df, "kinyarwanda", text_col,
        processor_or_tokenizer, model, device,
        batch_size, max_new_tokens, model_name,
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
        description="Translate a HuggingFace Dataset using NLLB or TranslateGemma on GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Path to input file (.csv, .parquet, .json) or HF dataset directory")
    parser.add_argument("output", help="Path to save the translated dataset")
    parser.add_argument("--text-col",       default="text",        help="Column with source text")
    parser.add_argument("--lang-col",       default="language",    help="Column with source language")
    parser.add_argument("--batch-size",     type=int, default=32,  help="Rows per GPU batch")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per row")
    parser.add_argument("--device",         type=int, default=0,   help="GPU index, -1 for CPU")
    parser.add_argument("--model",          default=DEFAULT_MODEL, help="NLLB or TranslateGemma model name")
    parser.add_argument("--format",         default="csv",         choices=["parquet", "csv", "json"],
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
    print(f"Model  : {args.model}  ({'TranslateGemma' if is_translategemma(args.model) else 'NLLB'})")
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