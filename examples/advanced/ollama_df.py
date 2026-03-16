import argparse
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = "gpt-oss:120b"
DEFAULT_OLLAMA   = "http://localhost:11434"
OLLAMA_TIMEOUT   = 1000        # seconds per request
MAX_RETRIES      = 3          # retry a failed row up to 3 times
RETRY_DELAY      = 5          # seconds to wait between retries

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
    p = Path(output_path)
    if not p.exists():
        return set()
    try:
        done_df = pd.read_csv(p)
        if "_original_index" in done_df.columns:
            # Only count rows that actually have a translation
            done_df = done_df[done_df["translation_text"].notna() & (done_df["translation_text"] != "")]
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
# Ollama client helpers
# ---------------------------------------------------------------------------

def check_ollama(base_url: str, model: str) -> None:
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
    # Explicit instruction to not think aloud — helps suppress verbose reasoning
    return (
        f"Translate the following text from {src_lang} to {tgt_lang}.\n"
        f"Return ONLY the translated text with no explanation, no preamble, "
        f"no reasoning, and no quotation marks. Do not include any thinking "
        f"or notes — output the translation immediately.\n\n"
        f"Text:\n{text}\n\n"
        f"Translation:"
    )


def extract_translation(data: dict) -> str:
    """
    Extract the final translation from an Ollama response dict.

    Thinking/reasoning models (like gpt-oss:120b) return:
      - data["thinking"] — the internal chain-of-thought (discard this)
      - data["response"] — the actual answer (use this)

    Also guards against done_reason="length", which means the model was
    cut off mid-output by num_predict. We raise so the retry logic can
    increase the budget and try again.
    """
    done_reason = data.get("done_reason", "stop")
    if done_reason == "length":
        raise ValueError(
            "Model hit num_predict limit (done_reason=length). "
            "Response may be truncated — will retry with higher budget."
        )

    # Prefer the `response` field (post-thinking answer)
    result = data.get("response", "").strip()

    # Fallback: if response is empty but thinking is present, the model may
    # have put the translation inside thinking by mistake — extract last
    # non-empty paragraph as a last resort
    if not result and data.get("thinking"):
        paragraphs = [p.strip() for p in data["thinking"].split("\n\n") if p.strip()]
        result = paragraphs[-1] if paragraphs else ""

    return result


def estimate_num_predict(text: str) -> int:
    """
    Estimate a safe num_predict budget based on input text length.

    Token budget breakdown:
      - Translation output  ≈ 1.5× input tokens  (target language is often wordier)
      - Thinking overhead   ≈ 2.0× input tokens  (residual even with think=False)
      - Safety buffer       ×  1.5

    Formula: ceil(word_count × 1.3 × 3.5 × 1.5) rounded up to nearest 512.
    Minimum of 512, maximum of 16384.
    """
    word_count   = len(text.split())
    input_tokens = word_count * 1.3          # words → tokens
    raw_budget   = input_tokens * 3.5 * 1.5  # translation + thinking + buffer
    # Round up to nearest 512 for clean values
    rounded      = max(512, int((raw_budget + 511) // 512) * 512)
    return min(rounded, 16384)               # cap at 16k to avoid runaway


def translate_one(
    idx: int,
    text: str,
    src_lang: str,
    tgt_lang: str,
    model: str,
    base_url: str,
) -> tuple[int, str]:
    """
    Translate a single row with automatic retries on failure.
    num_predict is sized dynamically from the input text length.
    On done_reason=length, doubles the budget and retries up to MAX_RETRIES.
    Raises the last exception if all retries are exhausted.
    """
    prompt      = build_prompt(text, src_lang, tgt_lang)
    base_budget = estimate_num_predict(text)

    # On truncation, double the budget up to 3 times before giving up
    token_budgets = [base_budget, base_budget * 2, base_budget * 4]
    token_budgets = [min(b, 16384) for b in token_budgets]  # cap each tier

    last_error = None
    attempt    = 0

    for budget in token_budgets:
        for retry in range(1, MAX_RETRIES + 1):
            attempt += 1
            payload = {
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "think":  False,       # top-level: respected by Ollama >= 0.6
                "options": {
                    "temperature": 0.0,
                    "num_predict": budget,
                    "think": False,    # also inside options for older versions
                },
            }
            try:
                with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
                    resp = client.post(f"{base_url}/api/generate", json=payload)
                    resp.raise_for_status()
                    data   = resp.json()
                    result = extract_translation(data)

                    if not result:
                        raise ValueError(f"Empty translation returned for idx={idx}")

                    return idx, result

            except ValueError as e:
                # done_reason=length — break inner retry loop, try bigger budget
                if "done_reason=length" in str(e) or "num_predict limit" in str(e):
                    last_error = e
                    break   # go to next budget tier
                # empty response — retry with same budget
                last_error = e
                if retry < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * retry)

            except Exception as e:
                last_error = e
                if retry < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * retry)

    raise RuntimeError(
        f"Row {idx} failed after {attempt} attempts "
        f"(budgets tried: {token_budgets}). Last error: {last_error}"
    )


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
    orig_indices = pending_df.index.tolist()
    total        = len(texts)

    print(f"\n[Ollama] [{src_language} → {tgt_language}]  {total:,} rows to translate  "
          f"|  concurrency {concurrency}  |  model {model!r}")

    results:  dict[int, str] = {}   # orig_index → translation
    failures: dict[int, str] = {}   # orig_index → error message

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
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
                orig_idx = future_to_orig_idx[future]
                try:
                    _, translation = future.result()
                    results[orig_idx] = translation

                    # Immediately flush to disk
                    row_data = pending_df.loc[orig_idx].to_dict()
                    row_data["translation_text"] = translation
                    row_data["model_name"]        = model
                    row_data["_original_index"]   = orig_idx
                    writer.write_row(row_data)

                except Exception as e:
                    # Log the failure but keep going — don't let one bad row
                    # crash the entire job
                    err_msg = str(e)
                    failures[orig_idx] = err_msg
                    pbar.write(f"  [FAIL] row {orig_idx}: {err_msg}")

    # ── Report failures clearly ────────────────────────────────────────────
    if failures:
        print(f"\n  ⚠  {len(failures):,} row(s) failed after {MAX_RETRIES} retries:")
        for orig_idx, err in failures.items():
            preview = str(pending_df.loc[orig_idx, text_col])[:80]
            print(f"     row {orig_idx:>6}: {err}  |  text: {preview!r}")
        print(f"  Re-run the script to retry — failed rows will be picked up automatically.\n")

    # Fill results back into the sub-dataframe
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
    print(f"Checking Ollama at {base_url} …")
    check_ollama(base_url, model)
    print("Ollama OK.\n")

    en_df, kin_df = split_by_language(df, lang_col)
    if "word_count" in list(en_df.columns):
        en_df = en_df.sort_values(by="word_count")
        kin_df = kin_df.sort_values(by="word_count")

    output_columns    = list(df.columns) + ["translation_text", "model_name", "_original_index"]
    writer            = IncrementalCSVWriter(output_path, output_columns)
    completed_indices = load_completed_indices(output_path)
    if completed_indices:
        print(f"Found {len(completed_indices):,} already-translated rows in {output_path} — will resume.\n")

    translated_en  = translate_language_df(
        en_df,  "english",     text_col, model, base_url,
        concurrency, writer, completed_indices,
    )
    translated_kin = translate_language_df(
        kin_df, "kinyarwanda", text_col, model, base_url,
        concurrency, writer, completed_indices,
    )

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
                        choices=["parquet", "csv", "json"],     help="Output format")
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