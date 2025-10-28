import os
import argparse

from datasets import load_dataset
from src.utils.dataset import extract_functions_from_patch

def main():
    parser = argparse.ArgumentParser(description='Build dataset from patches')
    parser.add_argument('--dataset', default='SWE-Gym/SWE-Gym', help='Input dataset path')
    parser.add_argument('--split', default='train', help='Dataset split to use')
    parser.add_argument('--output', required=True, help='Output file path for processed dataset')
    args = parser.parse_args()
    
    # Load and process dataset
    dataset = load_dataset(args.dataset, split=args.split).to_pandas()
    
    dataset["target"] = dataset.apply(
        lambda row: f"{extract_functions_from_patch(row["patch"])}", axis=1
    )

    # if output does not exist, create it
    output_dir = os.path.join(args.output, args.dataset.replace("/", "__") + "_" + args.split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "train.parquet")
    dataset.to_parquet(output_path)
    
if __name__ == "__main__":
    main()