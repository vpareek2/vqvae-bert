import numpy as np
import json
from pathlib import Path

def inspect_processed_data(file_path):
    """
    Inspect the processed numpy file containing code indices and adjectives.

    Args:
        file_path (str): Path to the .npy file
    """
    # Load the data
    data = np.load(file_path, allow_pickle=True)

    print(f"\nInspecting {file_path}")
    print("-" * 50)

    # Basic statistics
    print(f"Total number of samples: {len(data)}")

    # Inspect the first few samples
    print("\nFirst 3 samples:")
    for i in range(min(3, len(data))):
        sample = data[i]
        print(f"\nSample {i}:")
        print("Adjectives:", sample['adjectives'])
        print("Code Indices Shape:", sample['code_indices'].shape)
        print("Code Indices Range:",
              f"Min: {sample['code_indices'].min()}, "
              f"Max: {sample['code_indices'].max()}")

    # Aggregate statistics on adjectives
    all_adjectives = {}
    for sample in data:
        for category, adj in sample['adjectives'].items():
            if category not in all_adjectives:
                all_adjectives[category] = {}
            if adj not in all_adjectives[category]:
                all_adjectives[category][adj] = 0
            all_adjectives[category][adj] += 1

    print("\nAdjective Distribution:")
    for category, adjs in all_adjectives.items():
        print(f"\n{category.capitalize()} Adjectives:")
        # Sort by frequency
        sorted_adjs = sorted(adjs.items(), key=lambda x: x[1], reverse=True)
        for adj, count in sorted_adjs[:10]:  # Top 10
            print(f"{adj}: {count}")

    # Check code indices distribution
    all_code_indices = np.concatenate([sample['code_indices'].flatten() for sample in data])
    unique_codes = np.unique(all_code_indices)

    print("\nCode Indices Statistics:")
    print(f"Unique Code Indices: {len(unique_codes)}")
    print(f"Code Indices Range: Min {all_code_indices.min()}, Max {all_code_indices.max()}")

    # Frequency of code indices
    code_freq = np.bincount(all_code_indices)
    print("\nTop 10 Most Frequent Code Indices:")
    top_indices = np.argsort(code_freq)[-10:][::-1]
    for idx in top_indices:
        print(f"Code {idx}: {code_freq[idx]} times")

def main():
    # Paths to processed files
    base_path = Path('datasets')
    splits = ['train', 'val', 'test']

    for split in splits:
        file_path = base_path / f'processed_{split}.npy'
        inspect_processed_data(str(file_path))

if __name__ == '__main__':
    main()
