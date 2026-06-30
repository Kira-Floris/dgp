#!/usr/bin/env python3
"""
Translation Generation and Evaluation Script for Fleurs-MT-Benchmark
- Translates between English and Kinyarwanda using an NLLB (No Language Left Behind) model  # CHANGED: updated model description
- Evaluates translations using BLEU, chrF, and COMET metrics
- Outputs: CSV file with source, target, translation, and metric scores
"""

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # CHANGED: NLLB is seq2seq, not a chat pipeline
from datasets import load_dataset
from tqdm import tqdm
import argparse
from typing import List, Dict, Optional
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CHANGED: TranslationGenerator class — rewritten for NLLB:
#   1. Default model_name changed to "facebook/nllb-200-distilled-600M"
#      (any NLLB checkpoint works: 600M / 1.3B / 3.3B / 54B-MoE)
#   2. Replaced the chat "pipeline" entirely with
#      AutoModelForSeq2SeqLM + AutoTokenizer, since NLLB is a
#      pure encoder-decoder translation model, not instruction-tuned
#   3. translate_text() now sets tokenizer.src_lang and forces the
#      decoder to start with the FLORES-200 target language token,
#      instead of building a chat prompt
#   4. Added LANG_CODE_MAP to convert this script's "en"/"rw" codes
#      into FLORES-200 codes ("eng_Latn"/"kin_Latn") that NLLB expects
# ============================================================
class TranslationGenerator:
    """Handles translation generation using an NLLB seq2seq translation model"""

    # CHANGED: FLORES-200 codes required by NLLB. Add more language pairs here as needed.
    LANG_CODE_MAP = {
        "en": "eng_Latn",
        "rw": "kin_Latn",
    }

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cuda"):  # CHANGED: default model
        """
        Initialize the NLLB model and tokenizer

        Args:
            model_name: HuggingFace model identifier (any NLLB-200 checkpoint)
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model: {model_name} on {self.device}")

        # CHANGED: load tokenizer + seq2seq model directly instead of a chat pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 200,
        num_beams: int = 1  # CHANGED: configurable, default greedy (was hardcoded num_beams=5)
    ) -> str:
        """
        Translate a single text (kept for single-item use; translate_batch is the fast path)

        Args:
            text: Text to translate
            source_lang: Source language code ('en' or 'rw')
            target_lang: Target language code ('en' or 'rw')
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width (1 = greedy, fastest)

        Returns:
            Translated text
        """
        results = self.translate_batch_internal(
            [text], source_lang, target_lang, max_new_tokens, num_beams
        )
        return results[0] if results else ""

    def translate_batch_internal(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        num_beams: int
    ) -> List[str]:
        """
        CHANGED: New helper — runs a single forward pass over a list of texts
        (true batching) instead of one text at a time. This is what actually
        uses the GPU's parallelism.
        """
        src_flores = self.LANG_CODE_MAP.get(source_lang, source_lang)
        tgt_flores = self.LANG_CODE_MAP.get(target_lang, target_lang)

        try:
            self.tokenizer.src_lang = src_flores
            # CHANGED: padding=True + truncation=True lets the tokenizer batch
            # variable-length sentences together in one tensor
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_flores)

            # CHANGED: torch.inference_mode() instead of no_grad() — slightly faster,
            # disables autograd bookkeeping entirely
            with torch.inference_mode():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,  # CHANGED: was hardcoded 5, now configurable (default 1 = greedy)
                )

            translations = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            return [t.strip() for t in translations]
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return [""] * len(texts)

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 200,
        show_progress: bool = True,
        batch_size: int = 32,   # CHANGED: new param — number of sentences sent to the GPU per forward pass
        num_beams: int = 1      # CHANGED: new param — passed through to generate()
    ) -> List[str]:
        """
        Translate a batch of texts, processed in GPU-batched chunks
        (instead of one sentence at a time)

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            max_new_tokens: Maximum tokens to generate
            show_progress: Whether to show progress bar
            batch_size: How many sentences to translate per forward pass
            num_beams: Beam search width (1 = greedy, fastest; >1 = higher quality, slower)

        Returns:
            List of translated texts
        """
        translations = []

        # CHANGED: chunk texts into batches instead of iterating one at a time
        chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        iterator = tqdm(chunks, desc=f"Translating {source_lang} -> {target_lang}") if show_progress else chunks

        for chunk in iterator:
            chunk_translations = self.translate_batch_internal(
                chunk, source_lang, target_lang, max_new_tokens, num_beams
            )
            translations.extend(chunk_translations)

        return translations


class MetricEvaluator:
    """Handles evaluation metrics computation"""

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize metric evaluator

        Args:
            metrics: List of metrics to compute ['bleu', 'chrf', 'comet']
                    If None, all metrics are computed
        """
        self.metrics = metrics or ['bleu', 'chrf', 'comet']
        self.available_metrics = []

        # Import metrics module
        self._load_metrics_module()

        # Initialize metric objects
        self.metric_objects = {}
        self._initialize_metrics()

    def _load_metrics_module(self):
        """Load the metrics module from the provided code"""
        logger.info("Loading metrics module...")

        # Check if metrics module exists
        try:
            from dgp.metrics import (
                BLEUScore, chrFScore, COMETMetric,
                EvaluationInput, MetricResult
            )
            self.BLEUScore = BLEUScore
            self.chrFScore = chrFScore
            self.COMETMetric = COMETMetric
            self.EvaluationInput = EvaluationInput
            self.MetricResult = MetricResult
            logger.info("Metrics module loaded successfully")
        except ImportError as e:
            logger.error(f"Could not import metrics module: {e}")
            logger.error("Please ensure metrics.py is in the same directory")
            raise

    def _initialize_metrics(self):
        """Initialize metric computation objects"""
        if 'bleu' in self.metrics:
            try:
                self.metric_objects['bleu'] = self.BLEUScore(max_order=4)
                self.available_metrics.append('bleu')
                logger.info("BLEU metric initialized")
            except Exception as e:
                logger.warning(f"Could not initialize BLEU: {e}")

        if 'chrf' in self.metrics:
            try:
                self.metric_objects['chrf'] = self.chrFScore(word_order=2)
                self.available_metrics.append('chrf')
                logger.info("chrF++ metric initialized")
            except Exception as e:
                logger.warning(f"Could not initialize chrF: {e}")

        if 'comet' in self.metrics:
            logger.info("Initializing COMET metrics (this may take a while)...")
            try:
                # Initialize both English and Kinyarwanda COMET models
                # self.metric_objects['comet_en'] = self.COMETMetric(lang="kinyarwanda")
                self.metric_objects['comet_en'] = self.COMETMetric(lang="english")
                self.metric_objects['comet_rw'] = self.COMETMetric(lang="kinyarwanda")
                self.available_metrics.append('comet')
                logger.info("COMET metrics initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize COMET: {e}")
                logger.warning("COMET scores will not be computed")

    def evaluate_translation(
        self,
        original: str,
        translation: str,
        reference: str,
        direction: str,
        forward_translation: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single translation with all available metrics

        Args:
            original: Original source text
            translation: Generated translation (hypothesis)
            reference: Reference translation (target)
            direction: Translation direction ('en->rw' or 'rw->en')
            forward_translation: Forward translation (for COMET)

        Returns:
            Dictionary of metric scores
        """
        scores = {}

        # Create evaluation input for reference-based metrics (BLEU, chrF)
        # These compare translation against reference
        eval_input_ref = self.EvaluationInput(
            original_text=reference,
            back_translation=translation,
            forward_translation=forward_translation,
            source_lang=direction.split('->')[0],
            target_lang=direction.split('->')[1]
        )

        # BLEU score (translation vs reference)
        if 'bleu' in self.available_metrics:
            try:
                result = self.metric_objects['bleu'].compute(eval_input_ref)
                scores['bleu'] = result.score
            except Exception as e:
                logger.debug(f"BLEU computation error: {e}")
                scores['bleu'] = 0.0

        # chrF score (translation vs reference)
        if 'chrf' in self.available_metrics:
            try:
                result = self.metric_objects['chrf'].compute(eval_input_ref)
                scores['chrf'] = result.score
            except Exception as e:
                logger.debug(f"chrF computation error: {e}")
                scores['chrf'] = 0.0

        # COMET score (requires source, translation, reference)
        if 'comet' in self.available_metrics and forward_translation:
            try:
                # For COMET: src=original, mt=translation, ref=forward_translation
                eval_input_comet = self.EvaluationInput(
                    original_text=original,  # source
                    back_translation=translation,  # MT output
                    forward_translation=forward_translation,  # reference
                    source_lang=direction.split('->')[0],
                    target_lang=direction.split('->')[1]
                )

                # Choose appropriate COMET model based on target language
                comet_metric = self.metric_objects['comet_rw']
                # if direction == 'en->rw':
                #     comet_metric = self.metric_objects['comet_rw']
                # else:
                #     comet_metric = self.metric_objects['comet_en']

                result = comet_metric.compute(eval_input_comet)
                scores['comet'] = result.score
            except Exception as e:
                logger.debug(f"COMET computation error: {e}")
                scores['comet'] = 0.0

        return scores

    def evaluate_dataframe(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Evaluate all translations in a dataframe

        Args:
            df: DataFrame with columns: direction, source, target, translation
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with added metric columns
        """
        logger.info("Starting metric evaluation...")

        # Initialize metric columns
        for metric in self.available_metrics:
            if metric == 'comet':
                df['comet'] = 0.0
            else:
                df[metric] = 0.0

        # Process each row
        iterator = tqdm(df.iterrows(), total=len(df), desc="Computing metrics") if show_progress else df.iterrows()

        results = []
        for idx, row in iterator:
            # Get forward translation for COMET
            # For en->rw: forward is the translation, need to find rw->en translation
            # For rw->en: forward is the translation, need to find en->rw translation
            forward_translation = None
            if 'comet' in self.available_metrics:
                if row['direction'] == 'en->rw':
                    # Find corresponding rw->en translation using the reference
                    matching = df[(df['direction'] == 'rw->en') & (df['source'] == row['target'])]
                    if len(matching) > 0:
                        forward_translation = matching.iloc[0]['translation']
                else:  # rw->en
                    # Find corresponding en->rw translation using the reference
                    matching = df[(df['direction'] == 'en->rw') & (df['source'] == row['target'])]
                    if len(matching) > 0:
                        forward_translation = matching.iloc[0]['translation']

            # Evaluate
            scores = self.evaluate_translation(
                original=row['source'],
                translation=row['translation'],
                reference=row['target'],
                direction=row['direction'],
                forward_translation=forward_translation
            )

            results.append(scores)

        # Add scores to dataframe
        for metric in self.available_metrics:
            if metric == 'comet':
                df['comet'] = [r.get('comet', 0.0) for r in results]
            else:
                df[metric] = [r.get(metric, 0.0) for r in results]

        logger.info("Metric evaluation complete!")
        return df


def load_fleurs_dataset(dataset_name: str = "Kira-Floris/Fleurs-MT-Benchmark", split: str = "test"):
    """
    Load the Fleurs-MT-Benchmark dataset

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load

    Returns:
        Loaded dataset
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def generate_translations(
    dataset,
    translator: TranslationGenerator,
    output_file: str = "translations_output.csv",
    sample_size: int = None,
    max_new_tokens: int = 200,
    evaluate: bool = False,
    metrics: Optional[List[str]] = None,
    batch_size: int = 32,   # CHANGED: new param, passed through to translate_batch
    num_beams: int = 1      # CHANGED: new param, passed through to translate_batch
):
    """
    Generate bidirectional translations and optionally evaluate them

    Args:
        dataset: Input dataset with 'en' and 'rw' columns
        translator: TranslationGenerator instance
        output_file: Output CSV file path
        sample_size: Number of samples to process (None for all)
        max_new_tokens: Maximum tokens to generate
        evaluate: Whether to compute evaluation metrics
        metrics: List of metrics to compute (None for all)
        batch_size: Number of sentences translated per GPU forward pass
        num_beams: Beam search width (1 = greedy/fastest)
    """
    # Sample dataset if requested
    if sample_size:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        logger.info(f"Processing {len(dataset)} samples")

    # Extract English and Kinyarwanda texts
    english_texts = [item['en'] for item in dataset]
    kinyarwanda_texts = [item['rw'] for item in dataset]

    logger.info("=" * 60)
    logger.info("Direction 1: English -> Kinyarwanda")
    logger.info("=" * 60)

    # Translate English to Kinyarwanda
    en_to_rw = translator.translate_batch(
        english_texts,
        source_lang="en",
        target_lang="rw",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,   # CHANGED
        num_beams=num_beams      # CHANGED
    )

    logger.info("=" * 60)
    logger.info("Direction 2: Kinyarwanda -> English")
    logger.info("=" * 60)

    # Translate Kinyarwanda to English
    rw_to_en = translator.translate_batch(
        kinyarwanda_texts,
        source_lang="rw",
        target_lang="en",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,   # CHANGED
        num_beams=num_beams      # CHANGED
    )

    # Create DataFrames for both directions
    df_en_to_rw = pd.DataFrame({
        'direction': ['en->rw'] * len(english_texts),
        'source': english_texts,
        'target': kinyarwanda_texts,
        'translation': en_to_rw
    })

    df_rw_to_en = pd.DataFrame({
        'direction': ['rw->en'] * len(kinyarwanda_texts),
        'source': kinyarwanda_texts,
        'target': english_texts,
        'translation': rw_to_en
    })

    # Combine both directions
    df_combined = pd.concat([df_en_to_rw, df_rw_to_en], ignore_index=True)

    # Evaluate if requested
    if evaluate:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating translations...")
        logger.info("=" * 60)
        evaluator = MetricEvaluator(metrics=metrics)
        df_combined = evaluator.evaluate_dataframe(df_combined)

        # Print metric summaries
        print_metric_summary(df_combined)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_combined.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Translations saved to: {output_file}")
    logger.info(f"Total translations: {len(df_combined)} ({len(df_en_to_rw)} per direction)")

    # Print sample results
    logger.info("\n" + "=" * 60)
    logger.info("Sample translations:")
    logger.info("=" * 60)
    print("\nEnglish -> Kinyarwanda (first 2 examples):")
    print(df_en_to_rw.head(2).to_string(index=False))
    print("\nKinyarwanda -> English (first 2 examples):")
    print(df_rw_to_en.head(2).to_string(index=False))


def evaluate_existing_csv(
    input_csv: str,
    output_csv: Optional[str] = None,
    metrics: Optional[List[str]] = None
):
    """
    Evaluate an existing translation CSV file

    Args:
        input_csv: Path to input CSV with translations
        output_csv: Path to output CSV (if None, overwrites input)
        metrics: List of metrics to compute (None for all)
    """
    logger.info(f"Loading translations from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Validate required columns
    if 'source_text' in df.columns:
        df['source'] = df['source_text']
    if 'reference' in df.columns:
        df['target'] = df['reference']

    required_cols = ['direction', 'source', 'target', 'translation']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} translations")

    # Evaluate
    evaluator = MetricEvaluator(metrics=metrics)
    df_evaluated = evaluator.evaluate_dataframe(df)

    # Print metric summaries
    print_metric_summary(df_evaluated)

    # Save
    output_path = output_csv or input_csv
    df_evaluated.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Evaluated translations saved to: {output_path}")


def print_metric_summary(df: pd.DataFrame):
    """Print summary statistics for computed metrics"""
    logger.info("\n" + "=" * 60)
    logger.info("METRIC SUMMARY")
    logger.info("=" * 60)

    # Get metric columns
    metric_cols = [col for col in df.columns if col in ['bleu', 'chrf', 'comet']]

    if not metric_cols:
        logger.warning("No metric columns found in dataframe")
        return

    # Overall summary
    print("\n--- Overall Metrics ---")
    summary = df[metric_cols].describe().loc[['mean', 'std', 'min', 'max']]
    print(summary.to_string())

    # Per-direction summary
    print("\n--- Metrics by Direction ---")
    for direction in df['direction'].unique():
        print(f"\n{direction}:")
        direction_df = df[df['direction'] == direction]
        direction_summary = direction_df[metric_cols].describe().loc[['mean', 'std', 'min', 'max']]
        print(direction_summary.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate bidirectional translations for Fleurs-MT-Benchmark"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=['translate', 'evaluate', 'both'],
        default='both',
        help="Mode: 'translate' (generate translations), 'evaluate' (evaluate existing CSV), 'both' (do both)"
    )

    # Translation arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="Kira-Floris/Fleurs-MT-Benchmark",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/nllb-200-distilled-600M",    # CHANGED: any NLLB checkpoint, e.g. nllb-200-1.3B, nllb-200-3.3B
        help="Translation model name (NLLB-200 checkpoint)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fleurs_mt_benchmark_nllb_200_distilled_600m/en-rw.csv",  # CHANGED: results folder reflects NLLB model
        help="Output CSV file path"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to process (None for all)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    # CHANGED: new args for speed control
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sentences translated per GPU forward pass (higher = faster, more VRAM)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width (1 = greedy/fastest, >1 = higher quality but slower)"
    )

    # Evaluation arguments
    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/fleurs_mt_benchmark_nllb_200_distilled_600m/en-rw.csv",  # CHANGED: matches new --output default
        help="Input CSV file for evaluation mode"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        choices=['bleu', 'chrf', 'comet'],
        default='bleu chrf',
        help="Metrics to compute (default: all)"
    )
    parser.add_argument(
        "--no-evaluate",
        action='store_true',
        help="Skip evaluation when translating"
    )

    args = parser.parse_args()

    if args.mode == 'evaluate':
        # Evaluate existing CSV
        if not args.input_csv:
            parser.error("--input-csv is required for evaluate mode")
        evaluate_existing_csv(
            input_csv=args.input_csv,
            output_csv=args.output if args.output != "translations_output.csv" else None,
            metrics=args.metrics
        )

    elif args.mode == 'translate':
        # Generate translations only
        dataset = load_fleurs_dataset(args.dataset, args.split)
        translator = TranslationGenerator(model_name=args.model, device=args.device)
        generate_translations(
            dataset=dataset,
            translator=translator,
            output_file=args.output,
            sample_size=args.sample_size,
            max_new_tokens=args.max_tokens,
            evaluate=False,
            metrics=None,
            batch_size=args.batch_size,  # CHANGED
            num_beams=args.num_beams     # CHANGED
        )

    else:  # both
        # Generate translations and evaluate
        dataset = load_fleurs_dataset(args.dataset, args.split)
        translator = TranslationGenerator(model_name=args.model, device=args.device)
        generate_translations(
            dataset=dataset,
            translator=translator,
            output_file=args.output,
            sample_size=args.sample_size,
            max_new_tokens=args.max_tokens,
            evaluate=not args.no_evaluate,
            metrics=args.metrics,
            batch_size=args.batch_size,  # CHANGED
            num_beams=args.num_beams     # CHANGED
        )

    logger.info("Process complete!")


if __name__ == "__main__":
    main()
