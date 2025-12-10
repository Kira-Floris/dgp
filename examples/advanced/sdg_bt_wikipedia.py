# # from dgp.providers import GroqProvider
# # from dgp.providers import ModelConfig
# # from dgp.metrics import BLEUScore, COMETMetric
# # from dgp.tasks.backtranslation import BackTranslationPipeline

# # import os
# # from typing import List
# # import re

# # source_lang = "english"
# # intermediate_lang = "kinyarwanda"

# # data_dir = "data/wikimedia--wikipedia/20231101.simple/seed"

# # files = os.listdir(data_dir)

# # def split_into_sentences(text: str) -> List[str]:
# #     """
# #     Split text into sentences using regex patterns.
    
# #     Args:
# #         text: Input text to split
        
# #     Returns:
# #         List of sentences
# #     """
# #     sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
# #     sentences = re.split(sentence_endings, text)
# #     sentences = [s.strip() for s in sentences if s.strip()]
# #     return sentences

# # def get_metric_score(data: dict, metric_name: str):
# #     """
# #     Retrieve the score of a metric by name from a translation result dictionary.

# #     Args:
# #         data (dict): The dictionary containing translation results.
# #         metric_name (str): The name of the metric to retrieve, e.g. "COMET".

# #     Returns:
# #         float or None: The score if found, otherwise None.
# #     """
# #     metrics = data.get("metrics", [])
# #     for m in metrics:
# #         if m.get("name") == metric_name:
# #             return m.get("score")
# #     return None

# # """
# # algorithm
# # for each txt file in data dir
# # - create a temp file for it
# # - split the txt file content into sentences
# # - for each sentence in sentences
# #     for temperature between 0 and 1.0
# #         get a back translation for the sentence
# #         save the back translation results and scores to a list
# #     get the highest score at certain temperature
# #     save the original sentence, forward translation, comet score and index in sentence in a tsv file with same name as original txt file name
# #     go the the next sentence
# # - loop through the tsv file for each original sentence, and replace it in txt file and save it to a folder called data/wikimedia--wikipedia/20231101.simple/translation
# # """
# # provider = GroqProvider()
# # metrics = [BLEUScore(max_order=4), COMETMetric()]
# # model_name = "openai/gpt-oss-120b"
# # metric_name = "COMET"

# # temperature_values = [round(x * 0.1, 1) for x in range(0, 11)]
# # for file in files:
# #     with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
# #         text = f.read()
# #     sentences = split_into_sentences(text)
# #     results = []
# #     for sentence in sentences:
# #         for temperature in temperature_values:
# #             pipeline = BackTranslationPipeline(
# #                 provider=provider,
# #                 metrics=metrics,
# #                 model_config=ModelConfig(
# #                     model_name=model_name,
# #                     temperature=temperature
# #                 )
# #             )
# #             result = pipeline.run(
# #                 text=sentence,
# #                 source_lang=source_lang,
# #                 intermediate_lang=intermediate_lang,
# #                 system_template="Translate the following text from {src_lang} to {tgt_lang}. Return the translated text only."
# #             )
# #             results.append(result)
        
# #         highest = 0.0
# #         best_translation_result = None
# #         for result in results:
# #             score = get_metric_score(result, metric_name=metric_name)
# #             if score > highest:
# #                 highest = score
# #                 best_translation_result = result
        

# #         break
# #     break



# # # for sentence in sentences:
# # #     result = pipeline.run(
# # #         text=sentence,
# # #         source_lang=source_lang,
# # #         intermediate_lang=intermediate_lang,
# # #         system_template="Translate the following text from {src_lang} to {tgt_lang}. Return the translated text only."
# # #     )
# # #     print(result)

# # # print(result)

# from dgp.providers import GroqProvider, VLLMProvider
# from dgp.providers import ModelConfig
# from dgp.metrics import BLEUScore, COMETMetric
# from dgp.tasks.backtranslation import BackTranslationPipeline

# import os
# import csv
# from typing import List, Dict, Any
# import re
# from pathlib import Path
# from tqdm import tqdm
# import time

# source_lang = "english"
# intermediate_lang = "kinyarwanda"

# data_dir = "data/wikimedia--wikipedia/20231101.simple/seed"
# output_dir = "data/wikimedia--wikipedia/20231101.simple/translation"
# tsv_dir = "data/wikimedia--wikipedia/20231101.simple/tsv_results"

# # Create output directories if they don't exist
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(tsv_dir, exist_ok=True)

# files = os.listdir(data_dir)

# def split_into_sentences(text: str) -> List[str]:
#     """
#     Split text into sentences using regex patterns.
    
#     Args:
#         text: Input text to split
        
#     Returns:
#         List of sentences
#     """
#     sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
#     sentences = re.split(sentence_endings, text)
#     sentences = [s.strip() for s in sentences if s.strip()]
#     return sentences

# def get_metric_score(data: dict, metric_name: str):
#     """
#     Retrieve the score of a metric by name from a translation result dictionary.

#     Args:
#         data (dict): The dictionary containing translation results.
#         metric_name (str): The name of the metric to retrieve, e.g. "COMET".

#     Returns:
#         float or None: The score if found, otherwise None.
#     """
#     metrics = data.get("metrics", [])
#     for m in metrics:
#         if m.get("name") == metric_name:
#             return m.get("score")
#     return None

# def save_to_tsv(file_path: str, data: List[Dict[str, Any]]):
#     """
#     Save translation results to a TSV file.
    
#     Args:
#         file_path: Path to save the TSV file
#         data: List of dictionaries containing translation results
#     """
#     with open(file_path, 'w', newline='', encoding='utf-8') as f:
#         fieldnames = ['sentence_index', 'original_sentence', 'forward_translation', 
#                       'back_translation', 'comet_score', 'temperature']
#         writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
#         writer.writeheader()
#         writer.writerows(data)

# def load_from_tsv(file_path: str) -> List[Dict[str, Any]]:
#     """
#     Load translation results from a TSV file.
    
#     Args:
#         file_path: Path to the TSV file
        
#     Returns:
#         List of dictionaries containing translation results
#     """
#     results = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f, delimiter='\t')
#         for row in reader:
#             results.append(row)
#     return results

# def reconstruct_text(tsv_data: List[Dict[str, Any]]) -> str:
#     """
#     Reconstruct the text using forward translations from TSV data.
    
#     Args:
#         tsv_data: List of dictionaries with translation results
        
#     Returns:
#         Reconstructed text with forward translations
#     """
#     # Sort by sentence index to maintain order
#     sorted_data = sorted(tsv_data, key=lambda x: int(x['sentence_index']))
    
#     # Join forward translations with space
#     reconstructed = ' '.join(row['forward_translation'] for row in sorted_data)
    
#     return reconstructed


# # Main algorithm
# # provider = GroqProvider()
# provider = VLLMProvider()
# metrics = [BLEUScore(max_order=4), COMETMetric()]
# model_name = "openai/gpt-oss-20b"
# metric_name = "COMET"

# # temperature_values = [round(x * 0.1, 1) for x in range(0, 11)]
# temperature_values = [0.8, 0.9, 1.0]
# prompt = "Translate the following text from {src_lang} to {tgt_lang}. Return the translated text only."

# failure_attempts = 5

# print(f"Processing {len(files)} files...")
# print(f"Temperature range: {temperature_values}")
# print(f"Source language: {source_lang}")
# print(f"Intermediate language: {intermediate_lang}")
# print("=" * 80)

# for file_idx, file in tqdm(enumerate(files[:]), total=len(files), desc="Files Translated"):
#     print(f"\n[{file_idx + 1}/{len(files)}] Processing file: {file}")
    
#     file_path = os.path.join(data_dir, file)
#     output_path = os.path.join(output_dir, file)
#     if os.path.exists(output_path):
#         continue
    
#     # Read the original text
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()
    
#     # Split into sentences
#     sentences = split_into_sentences(text)
#     print(f"  Found {len(sentences)} sentences")
    
#     # Store best results for each sentence
#     best_results = []
    
#     # Process each sentence
#     for sent_idx, sentence in enumerate(sentences):
#         # print(f"  Sentence {sent_idx + 1}/{len(sentences)}: {sentence[:50]}...")
        
#         # Store all translation results for this sentence
#         temp_results = []

#         def runner(provider, metrics, model_name, temp, sentence, source_lang, intermediate_lang, prompt, attempt=0, num_attempts=failure_attempts):
#             pipeline = BackTranslationPipeline(
#                 provider=provider,
#                 metrics=metrics,
#                 model_config=ModelConfig(
#                     model_name=model_name,
#                     temperature=temp
#                 )
#             )
            
#             result = pipeline.run(
#                 text=sentence,
#                 source_lang=source_lang,
#                 intermediate_lang=intermediate_lang,
#                 system_template=prompt
#             )
#             if (result["forward"].strip() == "") or result["forward"] is None:
#                 if attempt <= num_attempts:
#                     time.sleep(2)
#                     return runner(provider, metrics, model_name, temp, sentence, source_lang, intermediate_lang, prompt, attempt+1)
#                 else:
#                     return result
#             return result
        
#         # Try different temperatures
#         for temp in temperature_values:
#             # print(f"    Temperature: {temp}", end=" ")
            
#             try:
#                 result = runner(provider, metrics, model_name, temp, sentence, source_lang, intermediate_lang, prompt, 0)
                
#                 score = get_metric_score(result, metric_name=metric_name)
#                 # print(f"Score: {score:.4f}")
                
#                 temp_results.append({
#                     'result': result,
#                     'temperature': temp,
#                     'score': score
#                 })
                
#             except Exception as e:
#                 print(f"ERROR: {str(e)}")
#                 continue
        
#         # Find the best translation (highest COMET score)
#         if temp_results:
#             best = max(temp_results, key=lambda x: x['score'] if x['score'] is not None else -1)
            
#             best_results.append({
#                 'sentence_index': sent_idx,
#                 'original_sentence': sentence,
#                 'forward_translation': best['result']['forward'],
#                 'back_translation': best['result']['back'],
#                 'comet_score': best['score'],
#                 'temperature': best['temperature']
#             })
            
#             # print(f"    ✓ Best score: {best['score']:.4f} at temperature {best['temperature']}")
#         else:
#             print(f"    ✗ No successful translations for this sentence")
    
#     # Save results to TSV
#     tsv_filename = Path(file).stem + '.tsv'
#     tsv_path = os.path.join(tsv_dir, tsv_filename)
#     save_to_tsv(tsv_path, best_results)
#     # print(f"  Saved TSV results to: {tsv_path}")
    
#     # Reconstruct text using forward translations
#     if best_results:
#         reconstructed_text = reconstruct_text(best_results)
        
#         # Save reconstructed text
#         output_filename = file
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             f.write(reconstructed_text)
        
#         # print(f"  Saved translated text to: {output_path}")
        
#         # Print statistics
#         avg_comet = sum(r['comet_score'] for r in best_results) / len(best_results)
#         min_comet = min(r['comet_score'] for r in best_results)
#         max_comet = max(r['comet_score'] for r in best_results)
        
#         # print(f"  Statistics:")
#         # print(f"    Avg COMET: {avg_comet:.4f}")
#         # print(f"    Min COMET: {min_comet:.4f}")
#         # print(f"    Max COMET: {max_comet:.4f}")
#     else:
#         print(f"  ✗ No results to save for this file")

# print("\n" + "=" * 80)
# print("Processing complete!")
# print(f"Translated files saved to: {output_dir}")
# print(f"TSV results saved to: {tsv_dir}")

from dgp.providers import GroqProvider, VLLMProvider
from dgp.providers import ModelConfig
from dgp.metrics import BLEUScore, COMETMetric, chrFScore
from dgp.tasks.backtranslation import BackTranslationPipeline

import os
import csv
from typing import List, Dict, Any
import re
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

source_lang = "english"
intermediate_lang = "kinyarwanda"

data_dir = "data/wikimedia--wikipedia/20231101.simple/seed"
output_dir = "data/wikimedia--wikipedia/20231101.simple/translation"
tsv_dir = "data/wikimedia--wikipedia/20231101.simple/tsv_results"

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tsv_dir, exist_ok=True)

files = os.listdir(data_dir)

# Thread-safe print with lock
print_lock = threading.Lock()

metrics = [
    BLEUScore(max_order=4), 
    # COMETMetric()
    chrFScore(word_order=2)
]

def thread_safe_print(*args, **kwargs):
    """Thread-safe printing"""
    with print_lock:
        print(*args, **kwargs)

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex patterns.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def get_metric_score(data: dict, metric_name: str):
    """
    Retrieve the score of a metric by name from a translation result dictionary.

    Args:
        data (dict): The dictionary containing translation results.
        metric_name (str): The name of the metric to retrieve, e.g. "COMET".

    Returns:
        float or None: The score if found, otherwise None.
    """
    metrics = data.get("metrics", [])
    for m in metrics:
        if m.get("name") == metric_name:
            return m.get("score")
    return None

def save_to_tsv(file_path: str, data: List[Dict[str, Any]]):
    """
    Save translation results to a TSV file.
    
    Args:
        file_path: Path to save the TSV file
        data: List of dictionaries containing translation results
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sentence_index', 'original_sentence', 'forward_translation', 
                      'back_translation', 'comet_score', 'temperature']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)

def load_from_tsv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load translation results from a TSV file.
    
    Args:
        file_path: Path to the TSV file
        
    Returns:
        List of dictionaries containing translation results
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            results.append(row)
    return results

def reconstruct_text(tsv_data: List[Dict[str, Any]]) -> str:
    """
    Reconstruct the text using forward translations from TSV data.
    
    Args:
        tsv_data: List of dictionaries with translation results
        
    Returns:
        Reconstructed text with forward translations
    """
    # Sort by sentence index to maintain order
    sorted_data = sorted(tsv_data, key=lambda x: int(x['sentence_index']))
    
    # Join forward translations with space
    reconstructed = ' '.join(row['forward_translation'] for row in sorted_data)
    
    return reconstructed

def runner(provider, metrics, model_name, temp, sentence, source_lang, intermediate_lang, prompt, attempt=0, num_attempts=5):
    """
    Run translation pipeline with retry logic.
    """
    pipeline = BackTranslationPipeline(
        provider=provider,
        metrics=metrics,
        model_config=ModelConfig(
            model_name=model_name,
            temperature=temp
        )
    )
    
    result = pipeline.run(
        text=sentence,
        source_lang=source_lang,
        intermediate_lang=intermediate_lang,
        system_template=prompt
    )
    
    if (result["forward"].strip() == "") or result["forward"] is None:
        if attempt <= num_attempts:
            time.sleep(2)
            return runner(provider, metrics, model_name, temp, sentence, source_lang, intermediate_lang, prompt, attempt+1, num_attempts)
        else:
            return result
    return result

def process_file(file: str, model_name: str, temperature_values: List[float], 
                 metric_name: str, prompt: str, failure_attempts: int) -> Dict[str, Any]:
    """
    Process a single file through the translation pipeline.
    
    Args:
        file: Filename to process
        model_name: Name of the model to use
        temperature_values: List of temperature values to try
        metric_name: Name of the metric to optimize for
        prompt: Translation prompt template
        failure_attempts: Number of retry attempts
        
    Returns:
        Dictionary with processing results and statistics
    """
    file_path = os.path.join(data_dir, file)
    output_path = os.path.join(output_dir, file)
    
    # Skip if already processed
    if os.path.exists(output_path):
        return {
            'file': file,
            'status': 'skipped',
            'message': 'Already processed'
        }
    
    try:
        # Create provider and metrics for this thread
        provider = VLLMProvider()
        # metrics = metrics
        global metrics
        
        # Read the original text
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        # Store best results for each sentence
        best_results = []
        
        # Process each sentence
        for sent_idx, sentence in enumerate(sentences):
            # Store all translation results for this sentence
            temp_results = []
            
            # Try different temperatures
            for temp in temperature_values:
                try:
                    result = runner(provider, metrics, model_name, temp, sentence, 
                                  source_lang, intermediate_lang, prompt, 0, failure_attempts)
                    
                    score = get_metric_score(result, metric_name=metric_name)
                    
                    temp_results.append({
                        'result': result,
                        'temperature': temp,
                        'score': score
                    })
                    
                except Exception as e:
                    thread_safe_print(f"ERROR in {file}, sentence {sent_idx}: {str(e)}")
                    continue
            
            # Find the best translation (highest COMET score)
            if temp_results:
                best = max(temp_results, key=lambda x: x['score'] if x['score'] is not None else -1)
                
                best_results.append({
                    'sentence_index': sent_idx,
                    'original_sentence': sentence,
                    'forward_translation': best['result']['forward'],
                    'back_translation': best['result']['back'],
                    'comet_score': best['score'],
                    'temperature': best['temperature']
                })
        
        # Save results to TSV
        tsv_filename = Path(file).stem + '.tsv'
        tsv_path = os.path.join(tsv_dir, tsv_filename)
        save_to_tsv(tsv_path, best_results)
        
        # Reconstruct text using forward translations
        if best_results:
            reconstructed_text = reconstruct_text(best_results)
            
            # Save reconstructed text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(reconstructed_text)
            
            # Calculate statistics
            avg_comet = sum(r['comet_score'] for r in best_results) / len(best_results)
            min_comet = min(r['comet_score'] for r in best_results)
            max_comet = max(r['comet_score'] for r in best_results)
            
            return {
                'file': file,
                'status': 'success',
                'sentences': len(sentences),
                'avg_comet': avg_comet,
                'min_comet': min_comet,
                'max_comet': max_comet
            }
        else:
            return {
                'file': file,
                'status': 'failed',
                'message': 'No results to save'
            }
            
    except Exception as e:
        return {
            'file': file,
            'status': 'error',
            'message': str(e)
        }

def main():
    """Main execution function with concurrent file processing"""
    
    # Configuration
    model_name = "openai/gpt-oss-20b"
    metric_name = "COMET"
    temperature_values = [1.0]
    prompt = "Translate the following text from {src_lang} to {tgt_lang}. Return the translated text only."
    failure_attempts = 5
    
    # Number of concurrent workers (adjust based on your system)
    max_workers = 4  # You can increase this based on your system resources
    
    print(f"Processing {len(files)} files...")
    print(f"Max concurrent workers: {max_workers}")
    print(f"Temperature range: {temperature_values}")
    print(f"Source language: {source_lang}")
    print(f"Intermediate language: {intermediate_lang}")
    print("=" * 80)
    
    results = []
    
    # Use ThreadPoolExecutor for concurrent file processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file processing tasks
        future_to_file = {
            executor.submit(
                process_file, 
                file, 
                model_name, 
                temperature_values, 
                metric_name, 
                prompt, 
                failure_attempts
            ): file 
            for file in files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(files), desc="Files Translated") as pbar:
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Print status for each completed file
                    if result['status'] == 'success':
                        thread_safe_print(
                            f"✓ {result['file']}: "
                            f"{result['sentences']} sentences, "
                            f"Avg COMET: {result['avg_comet']:.4f}"
                        )
                    elif result['status'] == 'skipped':
                        thread_safe_print(f"⊘ {result['file']}: {result['message']}")
                    else:
                        thread_safe_print(f"✗ {result['file']}: {result.get('message', 'Failed')}")
                    
                except Exception as e:
                    thread_safe_print(f"✗ {file}: Unexpected error - {str(e)}")
                    results.append({
                        'file': file,
                        'status': 'error',
                        'message': str(e)
                    })
                finally:
                    pbar.update(1)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"Translated files saved to: {output_dir}")
    print(f"TSV results saved to: {tsv_dir}")
    
    # Summary statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] in ['failed', 'error'])
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    
    # Overall statistics for successful files
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        overall_avg = sum(r['avg_comet'] for r in successful_results) / len(successful_results)
        print(f"  Overall Avg COMET: {overall_avg:.4f}")

if __name__ == "__main__":
    main()