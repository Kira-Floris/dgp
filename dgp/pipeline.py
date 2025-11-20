from typing import List, Dict, Tuple, Optional
import sacrebleu
from openai import OpenAI
import os
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .translate import translate_text
from .metrics import compute_chrf, compute_comet
from .languagedetection import detect_language

class SentenceLevelSyntheticTranslationPipeline:
    """
    Pipeline for generating synthetic parallel data for low-resource languages
    using iterative translation with temperature tuning.
    """
    
    def __init__(
        self, 
        translate_model: str,
        backtranslate_model: str,
        source_language: str = "English",
        target_language: str = "French",
        min_threshold: float = 50.0,
        max_tokens: int = 256,
        top_p: float = 0.95,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            translate_model: Model name for source -> target translation
            backtranslate_model: Model name for target -> source back-translation
            source_language: Source language name
            target_language: Target language name
            min_threshold: Minimum ChrF score to accept translations
            max_tokens: Maximum tokens for generation
            top_p: Top-p sampling parameter
            base_url: API base URL (for vLLM or local inference)
            api_key: OpenAI API key (if using OpenAI)
        """
        self.translate_model = translate_model
        self.backtranslate_model = backtranslate_model
        self.source_language = source_language
        self.target_language = target_language
        self.min_threshold = min_threshold
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.base_url = base_url
        self.api_key = api_key
    
    def translate(self, sentence: str, temperature: float) -> str:
        """
        Translate sentence using the translation model with given temperature.
        
        Args:
            sentence: Source sentence
            temperature: Sampling temperature
            
        Returns:
            Translated sentence
        """
        return translate_text(
            sentence=sentence,
            model=self.translate_model,
            source_language=self.source_language,
            target_language=self.target_language,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=self.top_p,
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def backtranslate(self, sentence: str, temperature: float) -> str:
        """
        Back-translate sentence using the back-translation model with given temperature.
        
        Args:
            sentence: Target sentence
            temperature: Sampling temperature
            
        Returns:
            Back-translated sentence
        """
        return translate_text(
            sentence=sentence,
            model=self.backtranslate_model,
            source_language=self.target_language,  # Swap languages for back-translation
            target_language=self.source_language,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=self.top_p,
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def process_sentence(self, sentence: str) -> Optional[Dict[str, any]]:
        """
        Process a single sentence through the pipeline.
        
        Args:
            sentence: Source sentence to process
            
        Returns:
            Dictionary with translation results or None if discarded
            {
                'source': original sentence,
                'translation': best translation,
                'chrf_score': best ChrF score,
                'temperature': temperature that produced best result
            }
        """
        # Initialize parameters
        k = 0.0
        best_score = 0.0
        best_translation = None
        best_backtranslation = None
        best_temperature = 0.0
        
        while k <= 1.0:
            # Translate using model with temperature k
            translation = self.translate(sentence, temperature=k)

            if detect_language(translation, lang_id="__label__kin_Latn") is False:
                # If language detection fails, skip this translation
                k = k + 0.1
                continue
            
            # Back-translate using model with temperature k
            backtranslation = self.backtranslate(translation, temperature=k)
            
            # Compute ChrF score
            # score = compute_chrf(sentence, backtranslation)
            score = compute_comet(translation, sentence, backtranslation)
        
            # Early stopping: if ChrF >= 95, save and return immediately
            if score >= 95.0:
                return {
                    'source': sentence,
                    'translation': translation,
                    'backtranslation': backtranslation,
                    'score': score,
                    'temperature': k,
                    'status': 'accepted_early_stop'
                }
            
            # Check if this is an improvement
            if score > best_score:
                best_score = score
                best_translation = translation
                best_backtranslation = backtranslation
                best_temperature = k
            
            # Increment temperature
            k = round(k + 0.1, 1)  # Round to avoid floating point errors
        
        # Check if best score meets minimum threshold
        if best_score >= self.min_threshold:
            return {
                'source': sentence,
                'translation': best_translation,
                'backtranslation': best_backtranslation,
                'score': best_score,
                'temperature': best_temperature,
                'status': 'accepted'
            }
        else:
            return None
    
    def process_batch(self, sentences: list) -> Tuple[list, list]:
        """
        Process a batch of sentences.
        
        Args:
            sentences: List of source sentences
            
        Returns:
            Tuple of (accepted_results, discarded_sentences)
        """
        accepted = []
        discarded = []
        
        for i, sentence in enumerate(sentences):
            
            result = self.process_sentence(sentence)
            
            if result is not None:
                accepted.append(result)
            else:
                discarded.append(sentence)
        
        return accepted, discarded


# @dataclass
# class TranslationRow:
#     """Represents a row with original text and its translation"""
#     row_id: int
#     original_text: str
#     sentences: List[str]
#     translated_sentences: List[Dict]
#     reconstructed_text: str
#     metadata: Dict


# class DocumentLevelSyntheticTranslationPipeline:
#     """
#     Complete pipeline that:
#     1. Takes rows of text
#     2. Splits into smaller sentences
#     3. Translates using the SyntheticDataPipeline
#     4. Reconstructs the translated text
#     5. Saves results
#     """
    
#     def __init__(
#         self,
#         translate_model: str,
#         backtranslate_model: str,
#         source_language: str = "English",
#         target_language: str = "French",
#         min_threshold: float = 50.0,
#         max_tokens: int = 256,
#         top_p: float = 0.95,
#         base_url: str = "http://localhost:8000/v1",
#         api_key: str = None,
#         output_dir: str = "translations",
#         output_filename: str = "translation_results.json"
#     ):
#         """Initialize the complete translation pipeline"""
#         self.translate_model = translate_model
#         self.backtranslate_model = backtranslate_model
#         self.source_language = source_language
#         self.target_language = target_language
#         self.min_threshold = min_threshold
#         self.max_tokens = max_tokens
#         self.top_p = top_p
#         self.base_url = base_url
#         self.api_key = api_key
        
#         # Initialize the translation pipeline
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True)
#         self.output_filename = output_filename
        
#     def split_into_sentences(self, text: str) -> List[str]:
#         """
#         Split text into sentences using regex patterns.
        
#         Args:
#             text: Input text to split
            
#         Returns:
#             List of sentences
#         """
#         # Common sentence ending patterns
#         # This regex looks for periods, exclamation marks, question marks
#         # followed by whitespace or end of string
#         sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
        
#         sentences = re.split(sentence_endings, text)
        
#         # Clean up sentences
#         sentences = [s.strip() for s in sentences if s.strip()]
        
#         return sentences
    
#     def reconstruct_text(
#         self, 
#         original_sentences: List[str], 
#         translated_results: List[Dict]
#     ) -> str:
#         """
#         Reconstruct the translated text, preserving structure where possible.
#         Uses the FORWARD TRANSLATION (source -> target), not the backtranslation.
#         The backtranslation was only used for quality checking via ChrF score.
        
#         Args:
#             original_sentences: Original sentence list
#             translated_results: List of translation result dictionaries
#                                Each dict has 'source' (original) and 'translation' (target language)
            
#         Returns:
#             Reconstructed translated text in the TARGET language
#         """
#         # Create a mapping of original sentences to their TARGET LANGUAGE translations
#         # Note: result['translation'] is the forward translation (source -> target)
#         #       The backtranslation (target -> source) was only used for ChrF scoring
#         translation_map = {
#             result['source']: result['translation']  # Keep the FORWARD translation
#             for result in translated_results
#         }
        
#         # Reconstruct by replacing original sentences with their translations
#         reconstructed_sentences = []
#         for sentence in original_sentences:
#             if sentence in translation_map:
#                 # Use the translation in the TARGET language
#                 reconstructed_sentences.append(translation_map[sentence])
#             else:
#                 # Keep original if translation failed quality check
#                 reconstructed_sentences.append(f"[UNTRANSLATED: {sentence}]")
        
#         # Join sentences with appropriate spacing
#         return " ".join(reconstructed_sentences)
    
#     def process_row(
#         self, 
#         row_id: int, 
#         text: str,
#     ) -> TranslationRow:
#         """
#         Process a single row through the complete pipeline.
        
#         Args:
#             row_id: Unique identifier for the row
#             text: Text to translate
            
#         Returns:
#             TranslationRow object with all processing details
#         """
        
#         # Step 1: Split into sentences
#         sentences = self.split_into_sentences(text)
        
#         # Step 2: Translate sentences
#         translation_pipeline = SentenceLevelSyntheticTranslationPipeline(
#             translate_model=self.translate_model,
#             backtranslate_model=self.backtranslate_model,
#             source_language=self.source_language,
#             target_language=self.target_language,
#             min_threshold=self.min_threshold,
#             max_tokens=self.max_tokens,
#             top_p=self.top_p,
#             base_url=self.base_url,
#             api_key=self.api_key
#         )

#         translated_results, discarded = translation_pipeline.process_batch(sentences)
        
#         # Step 3: Reconstruct text
#         reconstructed = self.reconstruct_text(sentences, translated_results)
        
#         # Create metadata
#         metadata = {
#             'total_sentences': len(sentences),
#             'translated_sentences': len(translated_results),
#             'discarded_sentences': len(discarded),
#             'success_rate': len(translated_results) / len(sentences) * 100 if sentences else 0,
#             'avg_score': sum(r['score'] for r in translated_results) / len(translated_results) if translated_results else 0
#         }
        
#         return TranslationRow(
#             row_id=row_id,
#             original_text=text,
#             sentences=sentences,
#             translated_sentences=translated_results,
#             reconstructed_text=reconstructed,
#             metadata=metadata
#         )
    
#     def process_rows(
#         self, 
#         texts: List[str],
        
#     ) -> List[TranslationRow]:
#         """
#         Process multiple rows of text and save each row immediately after processing.
        
#         Args:
#             texts: List of text strings to process
            
#         Returns:
#             List of TranslationRow objects
#         """
#         results = []
#         output_path = self.output_dir / self.output_filename

#         with ThreadPoolExecutor(max_workers=4) as executor:
#             futures = {executor.submit(self.process_row, i, text): i for i, text in enumerate(texts)}
#             for future in as_completed(futures):
#                 row_result = future.result()
#                 results.append(row_result)
                
#                 # Save immediately after processing each row
#                 serializable_results = [asdict(row) for row in results]
#                 with open(output_path, 'w', encoding='utf-8') as f:
#                     json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                
#                 print(f"✓ Saved row {row_result.row_id} to {output_path}")
        
#         return results
    
#     def save_results(
#         self, 
#         results: List[TranslationRow], 
#         filename: str = "translation_results.json"
#     ) -> Path:
#         """
#         Save translation results to JSON file.
        
#         Args:
#             results: List of TranslationRow objects
#             filename: Output filename
            
#         Returns:
#             Path to saved file
#         """
#         output_path = self.output_dir / filename
        
#         # Convert to serializable format
#         serializable_results = [asdict(row) for row in results]
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
#         print(f"\n✓ Results saved to: {output_path}")
#         return output_path
    

# def main():
#     """
#     Example usage of the complete translation pipeline.
#     """
    
#     # Initialize pipeline
#     # translation_model = "google/gemma-3-270m"
#     translation_model = "google/gemma-3-1b-it"
#     pipeline = DocumentLevelSyntheticTranslationPipeline(
#         translate_model=translation_model,
#         backtranslate_model=translation_model,
#         source_language="English",
#         target_language="French",
#         min_threshold=50.0,
#         base_url="http://localhost:8000/v1",
#         output_dir="translation_output"
#     )
    
#     # Sample data (rows of text)
#     sample_texts = [
#         """Hello, how are you? I hope you're having a great day. 
#         The weather is wonderful today and I'm feeling energetic.""",
        
#         """Machine translation has made significant progress in recent years. 
#         Neural networks have revolutionized the field. However, low-resource 
#         languages still face challenges.""",
        
#         """Learning a new language opens up new opportunities. It helps you 
#         understand different cultures. Practice is the key to success!"""
#     ]
    
#     # Process all rows
#     print("Starting translation pipeline...")
#     results = pipeline.process_rows(sample_texts)
    
#     # Save results
#     print("\n\nSaving results...")
#     # pipeline.save_results(results, "translation_results.json")
    
#     # Print summary
#     print("\n" + "="*70)
#     print("SUMMARY")
#     print("="*70)
    
#     total_sentences = sum(r.metadata['total_sentences'] for r in results)
#     total_translated = sum(r.metadata['translated_sentences'] for r in results)
#     avg_success_rate = sum(r.metadata['success_rate'] for r in results) / len(results)
    
#     print(f"Total rows processed: {len(results)}")
#     print(f"Total sentences: {total_sentences}")
#     print(f"Successfully translated: {total_translated}")
#     print(f"Average success rate: {avg_success_rate:.1f}%")
    
#     print("\n\nExample Output:")
#     print("="*70)
#     for i, result in enumerate(results[:2]):  # Show first 2 examples
#         print(f"\nRow {i}:")
#         print(f"Original: {result.original_text[:100]}...")
#         print(f"Translated: {result.reconstructed_text[:100]}...")


# if __name__ == "__main__":
#     from datetime import datetime
#     start_time = datetime.now()
#     main()
#     end_time = datetime.now()
#     print(f"\nTotal execution time: {end_time - start_time}")


# # 4:42
# # 4:04


from typing import List, Dict, Tuple, Optional
import sacrebleu
from openai import OpenAI
import os
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .translate import translate_text
from .metrics import compute_chrf


@dataclass
class TranslationRow:
    """Represents a row with original text and its translation"""
    row_id: int
    original_text: str
    sentences: List[str]
    translated_sentences: List[Dict]
    reconstructed_text: str
    metadata: Dict


class DocumentLevelSyntheticTranslationPipeline:
    """
    Complete pipeline that:
    1. Takes rows of text
    2. Splits into smaller sentences
    3. Translates using the SyntheticDataPipeline
    4. Reconstructs the translated text
    5. Saves results as individual row files
    """
    
    def __init__(
        self,
        translate_model: str,
        backtranslate_model: str,
        source_language: str = "English",
        target_language: str = "French",
        min_threshold: float = 50.0,
        max_tokens: int = 256,
        top_p: float = 0.95,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = None,
        output_dir: str = "translations",
        skip_existing: bool = True
    ):
        """
        Initialize the complete translation pipeline
        
        Args:
            skip_existing: If True, skip rows that have already been processed
        """
        self.translate_model = translate_model
        self.backtranslate_model = backtranslate_model
        self.source_language = source_language
        self.target_language = target_language
        self.min_threshold = min_threshold
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.base_url = base_url
        self.api_key = api_key
        self.skip_existing = skip_existing
        
        # Initialize the output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def get_row_filepath(self, row_id: int) -> Path:
        """
        Get the filepath for a specific row.
        
        Args:
            row_id: Row identifier
            
        Returns:
            Path to the row's JSON file
        """
        return self.output_dir / f"row_{row_id}.json"
    
    def is_row_processed(self, row_id: int) -> bool:
        """
        Check if a row has already been processed.
        
        Args:
            row_id: Row identifier
            
        Returns:
            True if the row file exists, False otherwise
        """
        return self.get_row_filepath(row_id).exists()
    
    def load_processed_row(self, row_id: int) -> Optional[TranslationRow]:
        """
        Load a previously processed row.
        
        Args:
            row_id: Row identifier
            
        Returns:
            TranslationRow object if file exists, None otherwise
        """
        filepath = self.get_row_filepath(row_id)
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TranslationRow(**data)
        except Exception as e:
            print(f"⚠ Warning: Could not load row {row_id}: {e}")
            return None
    
    def save_row(self, row: TranslationRow) -> Path:
        """
        Save a single row to its own JSON file.
        
        Args:
            row: TranslationRow object to save
            
        Returns:
            Path to the saved file
        """
        filepath = self.get_row_filepath(row.row_id)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(row), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def split_into_sentences(self, text: str) -> List[str]:
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
    
    def reconstruct_text(
        self, 
        original_sentences: List[str], 
        translated_results: List[Dict]
    ) -> str:
        """
        Reconstruct the translated text, preserving structure where possible.
        Uses the FORWARD TRANSLATION (source -> target), not the backtranslation.
        
        Args:
            original_sentences: Original sentence list
            translated_results: List of translation result dictionaries
            
        Returns:
            Reconstructed translated text in the TARGET language
        """
        translation_map = {
            result['source']: result['translation']
            for result in translated_results
        }
        
        reconstructed_sentences = []
        for sentence in original_sentences:
            if sentence in translation_map:
                reconstructed_sentences.append(translation_map[sentence])
            else:
                reconstructed_sentences.append(f"[UNTRANSLATED: {sentence}]")
        
        return " ".join(reconstructed_sentences)
    
    def process_row(
        self, 
        row_id: int, 
        text: str,
    ) -> TranslationRow:
        """
        Process a single row through the complete pipeline.
        
        Args:
            row_id: Unique identifier for the row
            text: Text to translate
            
        Returns:
            TranslationRow object with all processing details
        """
        # Import here to avoid circular dependency
        # from .pipeline import SentenceLevelSyntheticTranslationPipeline
        
        # Step 1: Split into sentences
        sentences = self.split_into_sentences(text)
        
        # Step 2: Translate sentences
        translation_pipeline = SentenceLevelSyntheticTranslationPipeline(
            translate_model=self.translate_model,
            backtranslate_model=self.backtranslate_model,
            source_language=self.source_language,
            target_language=self.target_language,
            min_threshold=self.min_threshold,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            base_url=self.base_url,
            api_key=self.api_key
        )

        translated_results, discarded = translation_pipeline.process_batch(sentences)
        
        # Step 3: Reconstruct text
        reconstructed = self.reconstruct_text(sentences, translated_results)
        
        # Create metadata
        metadata = {
            'total_sentences': len(sentences),
            'translated_sentences': len(translated_results),
            'discarded_sentences': len(discarded),
            'success_rate': len(translated_results) / len(sentences) * 100 if sentences else 0,
            'avg_score': sum(r['score'] for r in translated_results) / len(translated_results) if translated_results else 0
        }
        
        return TranslationRow(
            row_id=row_id,
            original_text=text,
            sentences=sentences,
            translated_sentences=translated_results,
            reconstructed_text=reconstructed,
            metadata=metadata
        )
    
    def process_rows(
        self, 
        texts: List[str],
        max_workers: int = 4
    ) -> List[TranslationRow]:
        """
        Process multiple rows of text, skipping already processed rows.
        Each row is saved immediately after processing.
        
        Args:
            texts: List of text strings to process
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of TranslationRow objects (including loaded and newly processed)
        """
        results = []
        rows_to_process = []
        
        # Check which rows need processing
        for i, text in enumerate(texts):
            if self.skip_existing and self.is_row_processed(i):
                # Load the existing row
                existing_row = self.load_processed_row(i)
                if existing_row:
                    results.append(existing_row)
                    print(f"⏭ Skipped row {i} (already processed)")
                else:
                    # File exists but couldn't load, reprocess
                    rows_to_process.append((i, text))
            else:
                rows_to_process.append((i, text))
        
        # Process remaining rows
        if rows_to_process:
            print(f"\n📝 Processing {len(rows_to_process)} new rows...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_row, row_id, text): row_id 
                    for row_id, text in rows_to_process
                }
                
                for future in as_completed(futures):
                    row_id = futures[future]
                    try:
                        row_result = future.result()
                        
                        # Save immediately after processing
                        filepath = self.save_row(row_result)
                        results.append(row_result)
                        
                        print(f"✓ Processed and saved row {row_id} to {filepath.name}")
                        
                    except Exception as e:
                        print(f"✗ Error processing row {row_id}: {e}")
        
        # Sort results by row_id for consistent ordering
        results.sort(key=lambda x: x.row_id)
        
        return results
    
    def aggregate_results(self, output_filename: str = "all_translations.json") -> Path:
        """
        Aggregate all individual row files into a single JSON file.
        
        Args:
            output_filename: Name for the aggregated output file
            
        Returns:
            Path to the aggregated file
        """
        # Find all row files
        row_files = sorted(self.output_dir.glob("row_*.json"))
        
        if not row_files:
            print("⚠ No row files found to aggregate")
            return None
        
        # Load all rows
        all_rows = []
        for filepath in row_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_rows.append(data)
            except Exception as e:
                print(f"⚠ Warning: Could not load {filepath.name}: {e}")
        
        # Sort by row_id
        all_rows.sort(key=lambda x: x['row_id'])
        
        # Save aggregated file
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Aggregated {len(all_rows)} rows to: {output_path}")
        return output_path
    
    def get_processing_stats(self) -> Dict:
        """
        Get statistics about processed rows.
        
        Returns:
            Dictionary with processing statistics
        """
        row_files = list(self.output_dir.glob("row_*.json"))
        
        if not row_files:
            return {
                'total_processed': 0,
                'files': []
            }
        
        stats = {
            'total_processed': len(row_files),
            'files': [f.name for f in sorted(row_files)]
        }
        
        return stats


def main():
    """
    Example usage of the complete translation pipeline with row-level files.
    """
    
    # Initialize pipeline
    translation_model = "google/gemma-3-1b-it"
    pipeline = DocumentLevelSyntheticTranslationPipeline(
        translate_model=translation_model,
        backtranslate_model=translation_model,
        source_language="English",
        target_language="Kinyarwanda",
        min_threshold=50.0,
        base_url="http://localhost:8000/v1",
        output_dir="translation_output",
        skip_existing=True  # Skip already processed rows
    )
    
    # Sample data (rows of text)
    sample_texts = [
        """Hello, how are you? I hope you're having a great day. 
        The weather is wonderful today and I'm feeling energetic.""",
        
        """Machine translation has made significant progress in recent years. 
        Neural networks have revolutionized the field. However, low-resource 
        languages still face challenges.""",
        
        """Learning a new language opens up new opportunities. It helps you 
        understand different cultures. Practice is the key to success!"""
    ]
    
    # Check existing processing status
    print("Checking processing status...")
    stats = pipeline.get_processing_stats()
    print(f"Already processed: {stats['total_processed']} rows")
    
    # Process all rows (will skip existing ones)
    print("\nStarting translation pipeline...")
    results = pipeline.process_rows(sample_texts, max_workers=4)
    
    # Aggregate results into single file (optional)
    print("\nAggregating results...")
    pipeline.aggregate_results("all_translations.json")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_sentences = sum(r.metadata['total_sentences'] for r in results)
    total_translated = sum(r.metadata['translated_sentences'] for r in results)
    avg_success_rate = sum(r.metadata['success_rate'] for r in results) / len(results) if results else 0
    
    print(f"Total rows processed: {len(results)}")
    print(f"Total sentences: {total_sentences}")
    print(f"Successfully translated: {total_translated}")
    print(f"Average success rate: {avg_success_rate:.1f}%")
    
    print("\n\nExample Output:")
    print("="*70)
    for i, result in enumerate(results[:2]):  # Show first 2 examples
        print(f"\nRow {i}:")
        print(f"Original: {result.original_text[:100]}...")
        print(f"Translated: {result.reconstructed_text[:100]}...")


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"\nTotal execution time: {end_time - start_time}")