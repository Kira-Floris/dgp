# from dgp.providers import GroqProvider, VLLMProvider
# from dgp.providers import ModelConfig
# from dgp.metrics import BLEUScore, COMETMetric, EvaluationInput
# from dgp.tasks.translation import TranslationPipeline

# from datasets import load_dataset
# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm

# def evaluate_translation(df, pipeline, source_lang, target_lang, source_col, target_col):
#     """
#     Evaluate translation quality for a given language pair.
    
#     Args:
#         df: DataFrame with test data
#         pipeline: Translation pipeline
#         source_lang: Source language name
#         target_lang: Target language name
#         source_col: Column name for source text
#         target_col: Column name for target (reference) text
    
#     Returns:
#         DataFrame with evaluation results
#     """
#     bleu_score = BLEUScore()
#     comet_score = COMETMetric()
    
#     results = []
    
#     print(f"\n{'='*80}")
#     print(f"Evaluating: {source_lang} → {target_lang}")
#     print(f"{'='*80}\n")
    
#     for index, row in tqdm(df.iterrows(), total=len(df), desc=f"{source_lang}→{target_lang}"):
#         text = row[source_col]
#         reference = row[target_col]
        
#         # Run translation
#         try:
#             result = pipeline.run(
#                 text=text,
#                 source_lang=source_lang.capitalize(),
#                 target_lang=target_lang.capitalize(),
#                 system_template="Translate the text from {src_lang} to {tgt_lang}. Return the translation only."
#             )
#         except Exception as e:
#             try:
#                 result = pipeline.run(
#                     text=text,
#                     source_lang=source_lang.capitalize(),
#                     target_lang=target_lang.capitalize(),
#                     system_template="Translate the text from {src_lang} to {tgt_lang}. Return the translation only."
#                 )
#             except Exception as e2:
#                 continue

        
#         translation = result.get("translation", "")
        
#         # Prepare evaluation input
#         eval_input = EvaluationInput(
#             original_text=reference,
#             back_translation=translation,
#             forward_translation=text,
#             source_lang=source_lang,
#             target_lang=target_lang
#         )
        
#         # Compute metrics
#         bleu = bleu_score.compute(eval_input)
#         comet = comet_score.compute(eval_input)
        
#         # Store results
#         results.append({
#             'index': index,
#             'source_text': text,
#             'reference': reference,
#             'translation': translation,
#             'bleu_score': bleu.score,
#             'comet_score': comet.score,
#             'source_lang': source_lang,
#             'target_lang': target_lang
#         })
        
#         # Print sample (first 3 examples)
#         if index < 3:
#             print(f"\nExample {index + 1}:")
#             print(f"Source: {text[:100]}...")
#             print(f"Reference: {reference[:100]}...")
#             print(f"Translation: {translation[:100]}...")
#             print(f"BLEU: {bleu.score}, COMET: {comet.score}")
#             print("-" * 80)
    
#     results_df = pd.DataFrame(results)
    
#     # Print summary statistics
#     print(f"\n{'='*80}")
#     print(f"Summary: {source_lang} → {target_lang}")
#     print(f"{'='*80}")
#     print(f"Total examples: {len(results_df)}")
#     print(f"Average BLEU: {results_df['bleu_score'].mean():.4f}")
#     print(f"Average COMET: {results_df['comet_score'].mean():.4f}")
#     print(f"BLEU Std Dev: {results_df['bleu_score'].std():.4f}")
#     print(f"COMET Std Dev: {results_df['comet_score'].std():.4f}")
#     print(f"{'='*80}\n")
    
#     return results_df


# def save_results(results_df, source_lang, target_lang, output_dir="results/fleurs_mt_benchmark"):
#     """
#     Save evaluation results to CSV.
    
#     Args:
#         results_df: DataFrame with evaluation results
#         source_lang: Source language name
#         target_lang: Target language name
#         output_dir: Directory to save results
#     """
#     # Create output directory
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # Create filename
#     filename = f"{source_lang}-{target_lang}.csv"
#     filepath = output_path / filename
    
#     # Save to CSV
#     results_df.to_csv(filepath, index=False)
#     print(f"✓ Results saved to: {filepath}")
    
#     return filepath


# if __name__ == "__main__":
#     # Load dataset
#     print("Loading FLEURS MT Benchmark dataset...")
#     dataset = load_dataset("Kira-Floris/Fleurs-MT-Benchmark")
#     df_test = dataset["test"].to_pandas()
#     print(f"Loaded {len(df_test)} test examples\n")
    
#     # Initialize pipeline
#     print("Initializing translation pipeline...")
#     pipeline = TranslationPipeline(
#         provider=VLLMProvider(),
#         model_config=ModelConfig(
#             model_name="google/gemma-3-4b-it",
#             temperature=0.0
#         )
#     )
#     print("Pipeline ready!\n")
    
#     # Define language pairs
#     language_pairs = [
#         ("english", "kinyarwanda", "en", "rw"),
#         ("kinyarwanda", "english", "rw", "en")
#     ]
    
#     # Evaluate each language pair
#     for source_lang, target_lang, source_col, target_col in language_pairs:
#         # Run evaluation
#         results_df = evaluate_translation(
#             df=df_test,
#             pipeline=pipeline,
#             source_lang=source_lang,
#             target_lang=target_lang,
#             source_col=source_col,
#             target_col=target_col
#         )
        
#         # Save results
#         save_results(results_df, source_col, target_col, output_dir="results/fleurs_mt_benchmark_gemma3_4b_base")
        
#         print("\n" + "="*80 + "\n")
    
#     print("✓ All evaluations complete!")
#     print(f"\nResults saved in: results/fleurs_mt_benchmark/")
#     print("  - english-kinyarwanda.csv")
#     print("  - kinyarwanda-english.csv")


from dgp.providers import GroqProvider, VLLMProvider
from dgp.providers import ModelConfig
from dgp.metrics import BLEUScore, COMETMetric, EvaluationInput
from dgp.tasks.translation import TranslationPipeline

from datasets import load_dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_translation(args):
    """
    Process a single translation example.
    
    Args:
        args: Tuple of (index, row, pipeline, bleu_score, comet_score, source_lang, target_lang, source_col, target_col)
    
    Returns:
        Dictionary with results or None if failed
    """
    index, row, pipeline, bleu_score, comet_score, source_lang, target_lang, source_col, target_col = args
    
    text = row[source_col]
    reference = row[target_col]
    
    # Run translation
    try:
        result = pipeline.run(
            text=text,
            source_lang=source_lang.capitalize(),
            target_lang=target_lang.capitalize(),
            system_template="Translate the text from {src_lang} to {tgt_lang}. Return the translation only."
        )
    except Exception as e:
        try:
            result = pipeline.run(
                text=text,
                source_lang=source_lang.capitalize(),
                target_lang=target_lang.capitalize(),
                system_template="Translate the text from {src_lang} to {tgt_lang}. Return the translation only."
            )
        except Exception as e2:
            return None
    
    translation = result.get("translation", "")
    
    # Prepare evaluation input
    eval_input = EvaluationInput(
        original_text=reference,
        back_translation=translation,
        forward_translation=text,
        source_lang=source_lang,
        target_lang=target_lang
    )
    
    # Compute metrics
    bleu = bleu_score.compute(eval_input)
    comet = comet_score.compute(eval_input)
    
    # Store results
    return {
        'index': index,
        'source_text': text,
        'reference': reference,
        'translation': translation,
        'bleu_score': bleu.score,
        'comet_score': comet.score,
        'source_lang': source_lang,
        'target_lang': target_lang
    }


def evaluate_translation(df, pipeline, bleu_score, comet_score, source_lang, target_lang, source_col, target_col, max_workers=4):
    """
    Evaluate translation quality for a given language pair in parallel.
    
    Args:
        df: DataFrame with test data
        pipeline: Translation pipeline
        bleu_score: BLEUScore instance
        comet_score: COMETMetric instance
        source_lang: Source language name
        target_lang: Target language name
        source_col: Column name for source text
        target_col: Column name for target (reference) text
        max_workers: Number of parallel workers
    
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {source_lang} → {target_lang}")
    print(f"{'='*80}\n")
    
    # Prepare arguments for parallel processing
    args_list = [
        (index, row, pipeline, bleu_score, comet_score, source_lang, target_lang, source_col, target_col)
        for index, row in df.iterrows()
    ]
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_translation, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{source_lang}→{target_lang}"):
            result = future.result()
            if result is not None:
                results.append(result)
                
                # Print sample (first 3 examples)
                if result['index'] < 3:
                    print(f"\nExample {result['index'] + 1}:")
                    print(f"Source: {result['source_text'][:100]}...")
                    print(f"Reference: {result['reference'][:100]}...")
                    print(f"Translation: {result['translation'][:100]}...")
                    print(f"BLEU: {result['bleu_score']}, COMET: {result['comet_score']}")
                    print("-" * 80)
    
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Summary: {source_lang} → {target_lang}")
    print(f"{'='*80}")
    print(f"Total examples: {len(results_df)}")
    print(f"Average BLEU: {results_df['bleu_score'].mean():.4f}")
    print(f"Average COMET: {results_df['comet_score'].mean():.4f}")
    print(f"BLEU Std Dev: {results_df['bleu_score'].std():.4f}")
    print(f"COMET Std Dev: {results_df['comet_score'].std():.4f}")
    print(f"{'='*80}\n")
    
    return results_df


def save_results(results_df, source_lang, target_lang, output_dir="results/fleurs_mt_benchmark"):
    """
    Save evaluation results to CSV.
    
    Args:
        results_df: DataFrame with evaluation results
        source_lang: Source language name
        target_lang: Target language name
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{source_lang}-{target_lang}.csv"
    filepath = output_path / filename
    
    # Save to CSV
    results_df.to_csv(filepath, index=False)
    print(f"✓ Results saved to: {filepath}")
    
    return filepath


if __name__ == "__main__":

    model_name = "Kira-Floris/Qwen3-4B"
    output_dir = "results/fleurs_mt_benchmark_kira_floris_qwen3_4b_stage2"

    # Load dataset
    print("Loading FLEURS MT Benchmark dataset...")
    dataset = load_dataset("Kira-Floris/Fleurs-MT-Benchmark")
    df_test = dataset["test"].to_pandas()
    print(f"Loaded {len(df_test)} test examples\n")
    
    # Initialize pipeline (loaded once)
    print("Initializing translation pipeline...")
    pipeline = TranslationPipeline(
        provider=VLLMProvider(),
        model_config=ModelConfig(
            model_name=model_name,
            temperature=0.0
        )
    )
    print("Pipeline ready!\n")
    
    # Initialize metrics (loaded once)
    print("Initializing metrics...")
    bleu_score = BLEUScore()
    comet_score = COMETMetric()
    print("Metrics ready!\n")
    
    # Define language pairs
    language_pairs = [
        ("english", "kinyarwanda", "en", "rw"),
        ("kinyarwanda", "english", "rw", "en")
    ]
    
    # Evaluate each language pair
    for source_lang, target_lang, source_col, target_col in language_pairs:
        # Run evaluation
        results_df = evaluate_translation(
            df=df_test,
            pipeline=pipeline,
            bleu_score=bleu_score,
            comet_score=comet_score,
            source_lang=source_lang,
            target_lang=target_lang,
            source_col=source_col,
            target_col=target_col,
            max_workers=4  # Adjust based on your system
        )
        
        # Save results
        save_results(results_df, source_col, target_col, output_dir=output_dir)
        
        print("\n" + "="*80 + "\n")
    
    print("✓ All evaluations complete!")
    print(f"\nResults saved in: results/fleurs_mt_benchmark/")
    print("  - english-kinyarwanda.csv")
    print("  - kinyarwanda-english.csv")