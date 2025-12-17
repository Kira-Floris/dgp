# python -m scripts.combined_tsv_score "data/wikimedia--wikipedia--gemma/20231101.simple/tsv_results" -o results.csv
# python -m scripts.combined_tsv_score "data/wikimedia--wikipedia/20231101.simple/tsv_results" -o results.csv
# python -m scripts.combined_tsv_score "data/mbazanlp--kinyarwanda_monolingual_v01.1/tsv_results" -o results.csv
# python -m scripts.combined_tsv_score "data/mbazanlp--kinyarwanda_monolingual_v01.1--gemma/tsv_results" -o results.csv


import os
import pandas as pd
from pathlib import Path
import argparse

def calculate_comet_average(folder_path, output_file=None, threshold=0.8):
    """
    Calculate the average comet_score across all TSV files in a folder.
    
    Args:
        folder_path: Path to folder containing TSV files
        output_file: Optional path to save results to CSV
        threshold: Score threshold to count scores above (default: 0.8)
    
    Returns:
        dict with results
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Find all TSV files
    tsv_files = list(folder.glob("*.tsv"))
    
    if not tsv_files:
        raise ValueError(f"No TSV files found in {folder_path}")
    
    print(f"Found {len(tsv_files)} TSV files")
    print("-" * 60)
    
    all_scores = []
    file_averages = []
    scores_above_threshold = 0
    threshold_by_file = []
    
    for tsv_file in tsv_files:
        try:
            # Read TSV file
            df = pd.read_csv(tsv_file, sep='\t')
            
            # Check if comet_score column exists
            if 'comet_score' not in df.columns:
                print(f"⚠️  Skipping {tsv_file.name}: No 'comet_score' column found")
                print(f"   Available columns: {', '.join(df.columns)}")
                continue
            
            # Get comet scores
            scores = df['comet_score'].dropna()
            
            if len(scores) == 0:
                print(f"⚠️  Skipping {tsv_file.name}: No valid comet_score values")
                continue
            
            # Calculate average for this file
            file_avg = scores.mean()
            file_averages.append({
                'file': tsv_file.name,
                'average_score': file_avg,
                'num_scores': len(scores),
                'min_score': scores.min(),
                'max_score': scores.max()
            })
            
            # Add all scores to global list
            all_scores.extend(scores.tolist())
            
            # print(f"✓ {tsv_file.name}")
            # print(f"  Average: {file_avg:.4f}")
            # print(f"  Count: {len(scores)}")
            # print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
            # print()
            
        except Exception as e:
            print(f"❌ Error reading {tsv_file.name}: {str(e)}")
            print()
    
    if not all_scores:
        raise ValueError("No valid comet_score data found in any TSV file")
    
    # Calculate overall average
    overall_avg = sum(all_scores) / len(all_scores)
    overall_min = min(all_scores)
    overall_max = max(all_scores)
    above_threshold_pct = (scores_above_threshold / len(all_scores)) * 100
    
    # Calculate score distribution
    score_ranges = {
        '0.0-0.2': sum(1 for s in all_scores if 0.0 <= s < 0.2),
        '0.2-0.4': sum(1 for s in all_scores if 0.2 <= s < 0.4),
        '0.4-0.6': sum(1 for s in all_scores if 0.4 <= s < 0.6),
        '0.6-0.8': sum(1 for s in all_scores if 0.6 <= s < 0.8),
        '0.8-1.0': sum(1 for s in all_scores if 0.8 <= s <= 1.0),
    }
    
    print("=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total files processed: {len(file_averages)}")
    print(f"Total scores: {len(all_scores)}")
    print(f"Overall average: {overall_avg:.4f}")
    print(f"Overall min: {overall_min:.4f}")
    print(f"Overall max: {overall_max:.4f}")
    print(f"Standard deviation: {pd.Series(all_scores).std():.4f}")
    print()
    print(f"Scores >= {threshold}: {scores_above_threshold:,} ({above_threshold_pct:.2f}%)")
    print(f"Scores < {threshold}: {len(all_scores) - scores_above_threshold:,} ({100 - above_threshold_pct:.2f}%)")
    print()
    print("Score Distribution:")
    for range_label, count in score_ranges.items():
        pct = (count / len(all_scores)) * 100
        bar = '█' * int(pct / 2)  # Scale bar to fit
        print(f"  {range_label}: {count:>7,} ({pct:>5.2f}%) {bar}")
    
    # Save to file if requested
    if output_file:
        results_df = pd.DataFrame(file_averages)
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    return {
        'overall_average': overall_avg,
        'total_scores': len(all_scores),
        'files_processed': len(file_averages),
        'overall_min': overall_min,
        'overall_max': overall_max,
        'std_dev': pd.Series(all_scores).std(),
        'threshold': threshold,
        'scores_above_threshold': scores_above_threshold,
        'scores_above_threshold_pct': above_threshold_pct,
        'score_distribution': score_ranges,
        'file_averages': file_averages
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate average comet_score from multiple TSV files'
    )
    parser.add_argument(
        'folder',
        help='Path to folder containing TSV files'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file for detailed results',
        default=None
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.8,
        help='Threshold to count scores above (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    try:
        results = calculate_comet_average(args.folder, args.output, args.threshold)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    
    # Example usage if run without arguments
    if len(sys.argv) == 1:
        print("Usage Examples:")
        print("  python script.py /path/to/tsv/folder")
        print("  python script.py /path/to/tsv/folder -o results.csv")
        print("  python script.py /path/to/tsv/folder -t 0.7  # Custom threshold")
        print("  python script.py /path/to/tsv/folder -t 0.8 -o results.csv")
        print("\nOr use in your code:")
        print("  from script import calculate_comet_average")
        print("  results = calculate_comet_average('/path/to/folder', threshold=0.8)")
        sys.exit(1)
    
    sys.exit(main())