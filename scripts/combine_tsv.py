# python -m scripts.combine_tsv "data/wikimedia--wikipedia--gemma/20231101.simple/tsv_results" -c -o results/gemma-3-27b/combined.csv -n 50 -s results/gemma-3-27b/combined_samples.csv --filter-threshold -t 0.8 
# python -m scripts.combine_tsv "data/wikimedia--wikipedia/20231101.simple/tsv_results" -c -o results/gpt-oss-20b/combined.csv -n 50 -s results/gpt-oss-20b/combined_samples.csv --filter-threshold -t 0.8
# python -m scripts.combine_tsv "data/mbazanlp--kinyarwanda_monolingual_v01.1/tsv_results" -c -o results/gpt-oss-20b__rw/combined.csv -n 50 -s results/gpt-oss-20b__rw/combined_samples.csv --filter-threshold -t 0.6

import os
import pandas as pd
from pathlib import Path
import argparse

def combine_tsv_files(folder_path, output_file="combined.tsv", sample_size=None, sample_output=None, 
                     threshold=None, score_column='comet_score'):
    """
    Combine all TSV files in a folder into a single TSV file, optionally creating a sample.
    
    Args:
        folder_path: Path to folder containing TSV files
        output_file: Path to save combined TSV file (default: combined.tsv)
        sample_size: Number of rows to sample (default: None)
        sample_output: Path to save sample TSV file (default: None, uses output_file with _sample suffix)
        threshold: Only sample rows with score >= threshold (default: None, no filtering)
        score_column: Column name to apply threshold to (default: 'comet_score')
    
    Returns:
        tuple: (combined_df, sample_df) - sample_df is None if no sampling requested
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
    
    all_dataframes = []
    
    for tsv_file in tsv_files:
        try:
            # Read TSV file
            df = pd.read_csv(tsv_file, sep='\t')
            
            # Add source file column
            df['source_file'] = tsv_file.name
            
            all_dataframes.append(df)
            # print(f"✓ {tsv_file.name}: {len(df)} rows")
            
        except Exception as e:
            print(f"❌ Error reading {tsv_file.name}: {str(e)}")
    
    if not all_dataframes:
        raise ValueError("No valid TSV files could be read")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    combined_df = combined_df.dropna()

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save combined file
    combined_df.to_csv(output_file, sep=',', index=False)
    
    print("=" * 60)
    print("COMBINATION COMPLETE")
    print("=" * 60)
    print(f"Total files combined: {len(all_dataframes)}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {', '.join(combined_df.columns)}")
    print(f"\n✓ Combined TSV saved to: {output_file}")
    
    # Create sample if requested
    sample_df = None
    if sample_size is not None:
        # Apply threshold filter if specified
        if threshold is not None:
            if score_column not in combined_df.columns:
                print(f"\n❌ Error: Column '{score_column}' not found in data")
                print(f"   Available columns: {', '.join(combined_df.columns)}")
                return combined_df, None
            
            # Filter for rows above threshold
            filtered_df = combined_df[combined_df[score_column] >= threshold].copy()
            
            print(f"\n📊 Threshold Filtering ({score_column} >= {threshold}):")
            print(f"   Rows above threshold: {len(filtered_df):,} ({(len(filtered_df)/len(combined_df)*100):.2f}%)")
            print(f"   Rows below threshold: {len(combined_df) - len(filtered_df):,} ({((len(combined_df)-len(filtered_df))/len(combined_df)*100):.2f}%)")
            
            if len(filtered_df) == 0:
                print(f"\n⚠️  Warning: No rows meet the threshold criteria ({score_column} >= {threshold})")
                return combined_df, None
            
            sampling_pool = filtered_df
            pool_description = f"rows with {score_column} >= {threshold}"
        else:
            sampling_pool = combined_df
            pool_description = "all rows"
        
        # Check if sample size is larger than available pool
        if sample_size > len(sampling_pool):
            print(f"\n⚠️  Warning: Sample size ({sample_size}) is larger than available {pool_description} ({len(sampling_pool)})")
            print(f"   Using all {len(sampling_pool)} rows instead")
            sample_df = sampling_pool.copy()
            actual_sample_size = len(sampling_pool)
        else:
            # Random sample from filtered pool
            sample_df = sampling_pool.sample(n=sample_size, random_state=42)
            actual_sample_size = sample_size

        if sample_output is not None:
            os.makedirs(os.path.dirname(sample_output), exist_ok=True)
        
        # Determine sample output filename
        if sample_output is None:
            output_path = Path(output_file)
            sample_output = output_path.parent / f"{output_path.stem}_sample{output_path.suffix}"
        
        # Save sample
        sample_df.to_csv(sample_output, sep=',', index=False)
        
        print("\n" + "=" * 60)
        print("SAMPLE CREATED")
        print("=" * 60)
        if threshold is not None:
            print(f"Sampling from: {pool_description}")
            print(f"Available pool size: {len(sampling_pool):,} rows")
        print(f"Sample size: {actual_sample_size:,} rows")
        print(f"Sample percentage of pool: {(actual_sample_size / len(sampling_pool)) * 100:.2f}%")
        print(f"Sample percentage of total: {(actual_sample_size / len(combined_df)) * 100:.2f}%")
        
        # Show score statistics if threshold was applied
        if threshold is not None and score_column in sample_df.columns:
            scores = sample_df[score_column].dropna()
            if len(scores) > 0:
                print(f"\n{score_column} statistics in sample:")
                print(f"  Mean: {scores.mean():.4f}")
                print(f"  Min: {scores.min():.4f}")
                print(f"  Max: {scores.max():.4f}")
                print(f"  Median: {scores.median():.4f}")
        
        print(f"\n✓ Sample TSV saved to: {sample_output}")
    
    return combined_df, sample_df


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
            
            # Count scores above threshold
            scores_above_threshold += sum(1 for s in scores if s >= threshold)
            
        except Exception as e:
            print(f"❌ Error reading {tsv_file.name}: {str(e)}")
    
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
        bar = '█' * int(pct / 2)
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
        description='Combine TSV files or calculate average comet_score'
    )
    parser.add_argument(
        'folder',
        help='Path to folder containing TSV files'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (CSV for stats, TSV for combine)',
        default=None
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.8,
        help='Threshold value (default: 0.8). Used for stats OR filtering samples when combining.'
    )
    parser.add_argument(
        '-c', '--combine',
        action='store_true',
        help='Combine TSV files into single TSV (instead of calculating stats)'
    )
    parser.add_argument(
        '-n', '--sample-size',
        type=int,
        default=None,
        help='Number of rows to sample from combined file (only with -c)'
    )
    parser.add_argument(
        '-s', '--sample-output',
        default=None,
        help='Output file for sample (default: adds _sample suffix to output file)'
    )
    parser.add_argument(
        '--filter-threshold',
        action='store_true',
        help='When sampling, only include rows with score >= threshold (use with -c and -n)'
    )
    parser.add_argument(
        '--score-column',
        default='comet_score',
        help='Column name for threshold filtering (default: comet_score)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.combine:
            # Combine mode
            output = args.output if args.output else "combined.tsv"
            threshold_for_sampling = args.threshold if args.filter_threshold else None
            combine_tsv_files(
                args.folder, 
                output, 
                args.sample_size, 
                args.sample_output,
                threshold_for_sampling,
                args.score_column
            )
        else:
            # Stats mode (original functionality)
            if args.sample_size:
                print("⚠️  Warning: --sample-size is only used with --combine flag")
            if args.filter_threshold:
                print("⚠️  Warning: --filter-threshold is only used with --combine flag")
            calculate_comet_average(args.folder, args.output, args.threshold)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    
    # Example usage if run without arguments
    if len(sys.argv) == 1:
        print("Usage Examples:")
        print("\nCombine TSV files:")
        print("  python script.py /path/to/tsv/folder -c")
        print("  python script.py /path/to/tsv/folder -c -o combined.tsv")
        print("\nCombine and create sample (all rows):")
        print("  python script.py /path/to/tsv/folder -c -n 1000")
        print("\nCombine and sample only high-scoring rows:")
        print("  python script.py /path/to/tsv/folder -c -n 1000 --filter-threshold")
        print("  python script.py /path/to/tsv/folder -c -n 1000 --filter-threshold -t 0.85")
        print("  python script.py /path/to/tsv/folder -c -n 500 --filter-threshold --score-column bleu_score -t 0.7")
        print("\nCalculate statistics:")
        print("  python script.py /path/to/tsv/folder")
        print("  python script.py /path/to/tsv/folder -o results.csv")
        print("  python script.py /path/to/tsv/folder -t 0.7")
        print("\nOr use in your code:")
        print("  from script import combine_tsv_files")
        print("  # Sample only rows with score >= 0.8")
        print("  combined_df, sample_df = combine_tsv_files('/path', 'out.tsv', sample_size=1000, threshold=0.8)")
        sys.exit(1)
    
    sys.exit(main())