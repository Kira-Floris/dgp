from openai import OpenAI
import os
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
import json
from datetime import datetime

PROMPT = """Translate the following text from {source_language} to {target_language}. Return ONLY the translation, nothing else. No explanations, no labels, no additional text.

Text: {sentence}

Translation:"""

def translate_text(
        sentence: str, 
        model: str="google/gemma-3-270m",
        source_language: str = "English",
        target_language: str = "French", 
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        base_url: str="http://localhost:8000/v1",
        api_key: str=None,
        max_retries: int = 3) -> str:
    
    if base_url:
        client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"  # vLLM doesn't require API key
        )
    else:
        client = OpenAI(api_key=api_key if api_key else os.getenv("OPENAI_API", None))
    
    global PROMPT
    prompt = PROMPT.format(
        source_language=source_language,
        target_language=target_language,
        sentence=sentence
    )
    
    last_error = None
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                # stop=["\n\n", "Text:", "Translation:"]
            )
            
            # Check if response has content
            if not response.choices:
                last_error = "No response choices"
                continue
            
            message_content = response.choices[0].message.content
            
            if message_content is None:
                last_error = "Empty response content"
                continue
            
            translation = message_content.strip()
            
            # Handle empty translations
            if not translation:
                last_error = "Empty translation"
                continue
            
            # Remove any potential labels that might have slipped through
            if translation.startswith(("Translation:", "translation:", "Answer:", "answer:")):
                translation = translation.split(":", 1)[1].strip()
            
            # Success! Return the translation
            return translation
        
        except Exception as e:
            last_error = str(e)
            # If it's the last attempt, we'll return the error below
            if attempt < max_retries - 1:
                continue
    
    # All retries failed
    return f"[ERROR after {max_retries} attempts: {last_error}]"


def evaluate_flores(
        model: str,
        source_lang: str,
        target_lang: str,
        base_url: str = "http://localhost:8000/v1",
        split: str = "devtest",
        max_samples: int = None,
        output_dir: str = "./results"):
    """
    Evaluate translation model on FLORES dataset
    
    Args:
        model: Model name/path on vLLM server
        source_lang: Source language code (e.g., 'eng_Latn' for English)
        target_lang: Target language code (e.g., 'kin_Latn' for Kinyarwanda)
        base_url: vLLM server URL
        split: Dataset split ('dev' or 'devtest')
        max_samples: Maximum number of samples to evaluate (None = all)
        output_dir: Directory to save results
    """
    
    # Language name mapping for prompt
    lang_names = {
        'eng_Latn': 'English',
        'kin_Latn': 'Kinyarwanda'
    }
    
    print(f"\n{'='*60}")
    print(f"Starting FLORES Evaluation")
    print(f"Model: {model}")
    print(f"Direction: {source_lang} → {target_lang}")
    print(f"Split: {split}")
    print(f"{'='*60}\n")
    
    # Load FLORES dataset
    print("Loading FLORES dataset...")
    try:
        dataset = load_dataset("facebook/flores", f"{source_lang}-{target_lang}", split=split)
    except:
        # Try alternative loading method
        dataset = load_dataset("facebook/flores", name=f"{source_lang}-{target_lang}", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples\n")
    
    # Prepare containers
    hypotheses = []
    references = []
    sources = []
    error_count = 0
    
    # Get column names (they vary in FLORES)
    src_col = f"sentence_{source_lang}"
    tgt_col = f"sentence_{target_lang}"
    
    # Translate each sentence
    print("Translating sentences...")
    for idx, example in enumerate(tqdm(dataset)):
        source_text = example[src_col]
        reference_text = example[tgt_col]
        
        # Translate
        translation = translate_text(
            sentence=source_text,
            model=model,
            source_language=lang_names[source_lang],
            target_language=lang_names[target_lang],
            base_url=base_url
        )
        
        # Track errors
        if translation.startswith("[ERROR:"):
            error_count += 1
            if error_count <= 5:  # Print first 5 errors for debugging
                print(f"\n⚠️  Error at sample {idx + 1}:")
                print(f"   Source: {source_text[:80]}...")
                print(f"   Error: {translation}")
        
        sources.append(source_text)
        hypotheses.append(translation)
        references.append(reference_text)
        
        # Print sample every 100 sentences
        if (idx + 1) % 100 == 0:
            print(f"\n--- Sample {idx + 1} ---")
            print(f"Source: {source_text[:100]}...")
            print(f"Translation: {translation[:100]}...")
            print(f"Reference: {reference_text[:100]}...")
            print(f"Errors so far: {error_count}\n")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    bleu = BLEU()
    chrf = CHRF()
    
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])
    
    # Prepare results
    results = {
        "model": model,
        "source_language": source_lang,
        "target_language": target_lang,
        "split": split,
        "num_samples": len(dataset),
        "num_errors": error_count,
        "error_rate": error_count / len(dataset) if len(dataset) > 0 else 0,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "BLEU": {
                "score": bleu_score.score,
                "precisions": bleu_score.precisions,
                "bp": bleu_score.bp,
                "sys_len": bleu_score.sys_len,
                "ref_len": bleu_score.ref_len
            },
            "chrF": {
                "score": chrf_score.score
            }
        }
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"BLEU Score: {bleu_score.score:.2f}")
    print(f"chrF Score: {chrf_score.score:.2f}")
    print(f"Total Errors: {error_count} / {len(dataset)} ({error_count/len(dataset)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    direction = f"{source_lang}-{target_lang}"
    metrics_file = os.path.join(output_dir, f"metrics_{direction}_{split}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")
    
    # Save translations
    translations_file = os.path.join(output_dir, f"translations_{direction}_{split}.jsonl")
    with open(translations_file, 'w', encoding='utf-8') as f:
        for src, hyp, ref in zip(sources, hypotheses, references):
            f.write(json.dumps({
                "source": src,
                "translation": hyp,
                "reference": ref
            }, ensure_ascii=False) + '\n')
    print(f"Translations saved to: {translations_file}\n")
    
    return results


def evaluate_bidirectional(
        model: str,
        base_url: str = "http://localhost:8000/v1",
        split: str = "devtest",
        max_samples: int = None,
        output_dir: str = "./results"):
    """
    Evaluate both English→Kinyarwanda and Kinyarwanda→English
    """
    
    results = {}
    
    # English → Kinyarwanda
    print("\n" + "="*60)
    print("EVALUATING: English → Kinyarwanda")
    print("="*60)
    results['eng_to_kin'] = evaluate_flores(
        model=model,
        source_lang='eng_Latn',
        target_lang='kin_Latn',
        base_url=base_url,
        split=split,
        max_samples=max_samples,
        output_dir=output_dir
    )
    
    # Kinyarwanda → English
    print("\n" + "="*60)
    print("EVALUATING: Kinyarwanda → English")
    print("="*60)
    results['kin_to_eng'] = evaluate_flores(
        model=model,
        source_lang='kin_Latn',
        target_lang='eng_Latn',
        base_url=base_url,
        split=split,
        max_samples=max_samples,
        output_dir=output_dir
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - BIDIRECTIONAL EVALUATION")
    print("="*60)
    print(f"English → Kinyarwanda:")
    print(f"  BLEU: {results['eng_to_kin']['metrics']['BLEU']['score']:.2f}")
    print(f"  chrF: {results['eng_to_kin']['metrics']['chrF']['score']:.2f}")
    print(f"\nKinyarwanda → English:")
    print(f"  BLEU: {results['kin_to_eng']['metrics']['BLEU']['score']:.2f}")
    print(f"  chrF: {results['kin_to_eng']['metrics']['chrF']['score']:.2f}")
    print("="*60 + "\n")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"summary_bidirectional_{split}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")
    
    return results


if __name__ == "__main__":
    # Configuration
    MODEL = "openai/gpt-oss-120b"  # Change to your model
    BASE_URL = "http://localhost:8000/v1"
    SPLIT = "devtest"  # or "dev"
    MAX_SAMPLES = None  # Set to a number for testing (e.g., 100)
    OUTPUT_DIR = "./dgp/evaluation/results/flores_results"
    
    # Run bidirectional evaluation
    results = evaluate_bidirectional(
        model=MODEL,
        base_url=BASE_URL,
        split=SPLIT,
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR
    )