import argparse
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = "gpt-oss:120b"           # OpenAI open-weight model via Ollama
DEFAULT_OLLAMA   = "http://localhost:11434"  # Ollama default base URL
OLLAMA_TIMEOUT   = 300                       # seconds per request

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
    """
    Read already-translated rows from a previous (possibly partial) run.
    Returns a set of original DataFrame indices that are already done,
    so we can skip them and resume where we left off.
    """
    p = Path(output_path)
    if not p.exists():
        return set()
    try:
        done_df = pd.read_csv(p)
        if "_original_index" in done_df.columns:
            return set(done_df["_original_index"].tolist())
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Incremental writer
# ---------------------------------------------------------------------------

class IncrementalCSVWriter:
    """
    Thread-safe CSV writer that appends one row at a time to the output file.
    On the first write it creates the file and writes the header.
    Subsequent writes append rows without re-writing the header.
    A threading.Lock ensures two threads never write simultaneously.
    """

    def __init__(self, output_path: str, columns: list[str]) -> None:
        self.path    = Path(output_path)
        self.columns = columns
        self.lock    = threading.Lock()
        self._initialised = False

        # Create parent directories if needed
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # If the file already exists (resume run), mark as already initialised
        if self.path.exists():
            self._initialised = True

    def write_row(self, row: dict) -> None:
        """Append a single row dict to the CSV file (thread-safe)."""
        with self.lock:
            file_exists = self.path.exists() and self._initialised
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                if not file_exists:
                    writer.writeheader()
                    self._initialised = True
                writer.writerow({col: row.get(col, "") for col in self.columns})


# ---------------------------------------------------------------------------
# Ollama client helpers
# ---------------------------------------------------------------------------

def check_ollama(base_url: str, model: str) -> None:
    """Verify Ollama is running and the model is available."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from e

    available  = [m["name"] for m in resp.json().get("models", [])]
    base_names = [n.split(":")[0] for n in available]
    if model.split(":")[0] not in base_names:
        print(
            f"[WARN] Model {model!r} not found in Ollama. "
            f"Available: {available}\n"
            f"Pull it with: ollama pull {model}"
        )


def build_prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    return (
        f"Translate the following text from {src_lang} to {tgt_lang}.\n"
        f"Return ONLY the translated text with no explanation, no preamble, "
        f"and no quotation marks.\n\n"
        f"Text:\n{text}\n\n"
        f"Translation:"
    )


def translate_one(
    idx: int,
    text: str,
    src_lang: str,
    tgt_lang: str,
    model: str,
    base_url: str,
) -> tuple[int, str]:
    """
    Send a single synchronous translation request to Ollama.
    Returns (original_index, translation) so the caller can restore row order.
    Each thread gets its own httpx.Client for connection-pool isolation.
    """
    prompt  = build_prompt(text, src_lang, tgt_lang)
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
        },
    }
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        resp = client.post(f"{base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return idx, resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Step 1 — Split
# ---------------------------------------------------------------------------

def split_by_language(
    df: pd.DataFrame,
    lang_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
# Step 2 — Translate via Ollama (ThreadPoolExecutor) with incremental saving
# ---------------------------------------------------------------------------

def translate_language_df(
    sub_df: pd.DataFrame,
    src_language: str,
    text_col: str,
    model: str,
    base_url: str,
    concurrency: int,
    writer: IncrementalCSVWriter,
    completed_indices: set[int],
) -> pd.DataFrame:
    tgt_language = TRANSLATION_PAIR[src_language.lower()]
    src_name     = LANGUAGE_NAMES[src_language.lower()]
    tgt_name     = LANGUAGE_NAMES[tgt_language]

    sub_df = sub_df.copy()
    sub_df["translation_text"] = ""
    sub_df["model_name"]       = model

    if sub_df.empty:
        return sub_df

    # ── Resume: skip rows already saved in a previous run ─────────────────
    pending_df = sub_df[~sub_df.index.isin(completed_indices)]
    skipped    = len(sub_df) - len(pending_df)
    if skipped:
        print(f"  Resuming — skipping {skipped:,} already-translated rows")

    if pending_df.empty:
        print(f"  All {len(sub_df):,} rows already translated, nothing to do.")
        return sub_df

    texts        = pending_df[text_col].tolist()
    orig_indices = pending_df.index.tolist()   # real DataFrame indices
    total        = len(texts)

    print(f"\n[Ollama] [{src_language} → {tgt_language}]  {total:,} rows to translate  "
          f"|  concurrency {concurrency}  |  model {model!r}")

    results: dict[int, str] = {}   # orig_index → translation

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Map each Future to its original DataFrame index
        future_to_orig_idx = {
            executor.submit(
                translate_one,
                orig_idx, text, src_name, tgt_name, model, base_url,
            ): orig_idx
            for orig_idx, text in zip(orig_indices, texts)
        }

        with tqdm(
            as_completed(future_to_orig_idx),
            total=total,
            desc=f"  {src_language[:2].upper()}→{tgt_language[:2].upper()}",
            unit="row",
            dynamic_ncols=True,
            colour="green",
        ) as pbar:
            for future in pbar:
                orig_idx, translation = future.result()
                results[orig_idx] = translation

                # ── Immediately flush this row to disk ─────────────────────
                row_data = pending_df.loc[orig_idx].to_dict()
                row_data["translation_text"]  = translation
                row_data["model_name"]        = model
                row_data["_original_index"]   = orig_idx   # used for resume
                writer.write_row(row_data)

    # Fill translated results back into the sub-dataframe
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
    model: str,
    base_url: str,
    concurrency: int,
    output_path: str,
) -> pd.DataFrame:
    # ── 1. Verify Ollama connection ───────────────────────────────────────
    print(f"Checking Ollama at {base_url} …")
    check_ollama(base_url, model)
    print("Ollama OK.\n")

    # ── 2. Split ──────────────────────────────────────────────────────────
    en_df, kin_df = split_by_language(df, lang_col)

    # ── 3. Set up incremental writer + detect already-done rows ───────────
    output_columns = list(df.columns) + ["translation_text", "model_name", "_original_index"]
    writer             = IncrementalCSVWriter(output_path, output_columns)
    completed_indices  = load_completed_indices(output_path)
    if completed_indices:
        print(f"Found {len(completed_indices):,} already-translated rows in {output_path} — will resume.\n")

    # ── 4. Translate each sub-dataframe, writing each row as it completes ─
    translated_en  = translate_language_df(
        en_df,  "english",     text_col, model, base_url,
        concurrency, writer, completed_indices,
    )
    translated_kin = translate_language_df(
        kin_df, "kinyarwanda", text_col, model, base_url,
        concurrency, writer, completed_indices,
    )

    # ── 5. Recombine for the return value (output file is already written) ─
    result = recombine(translated_en, translated_kin)
    print(f"Recombined → {len(result):,} rows total")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a dataset (English ↔ Kinyarwanda) via a local Ollama model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Path to input file (.csv, .parquet, .json) or HF dataset dir")
    parser.add_argument("output", help="Path to save the translated dataset")
    parser.add_argument("--text-col",    default="text",        help="Column with source text")
    parser.add_argument("--lang-col",    default="language",    help="Column with source language")
    parser.add_argument("--model",       default=DEFAULT_MODEL, help="Ollama model name (must be pulled)")
    parser.add_argument("--ollama-url",  default=DEFAULT_OLLAMA,help="Ollama base URL")
    parser.add_argument("--concurrency", type=int, default=4,   help="Max parallel Ollama requests")
    parser.add_argument("--format",      default="csv",
                        choices=["parquet", "csv", "json"],     help="Output format (incremental saving is CSV only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Model       : {args.model}")
    print(f"Ollama URL  : {args.ollama_url}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Input       : {args.input}")
    print(f"Output      : {args.output}  [incremental CSV]")

    df = load_dataset(args.input)
    print(f"Loaded {len(df):,} rows")

    run_pipeline(
        df,
        text_col=args.text_col,
        lang_col=args.lang_col,
        model=args.model,
        base_url=args.ollama_url,
        concurrency=args.concurrency,
        output_path=args.output,
    )

    print(f"\nDone. Results saved incrementally to: {args.output}")


if __name__ == "__main__":
    main()