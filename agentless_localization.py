import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                results.append(json.loads(line))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read and parse Agentless localization results"
    )
    parser.add_argument(
        "--input_file", type=str, help="Path to the input JSONL file"
    )
    parser.add_argument(
        "--num_samples", type=int, help="Number of samples to print"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}")
        exit(1)

    all_results = read_jsonl(args.input_file)

    num_samples = args.num_samples if args.num_samples else len(all_results)

    for sample_results in all_results[:num_samples]:
        print(f"\nInstance ID: {sample_results['instance_id']}")
        print(
            f"\nFound Files: \n\t{'\n\t'.join(sample_results['found_files'])}"
        )

        print("\nFound Related Locs:")
        for file_path, related_locs in sample_results[
            "found_related_locs"
        ].items():
            print(f"\n\tFile Path: {file_path}")
            print("\tRelated Locs:")
            for loc in related_locs:
                print(f"\t\t{'\n\t\t'.join(loc.split('\n'))}")

        print("\nFound Edit Locs:")
        for idx, edit_loc_dict in enumerate(
            sample_results["found_edit_locs"], 1
        ):
            print(f"\n\tEdit Location Set #{idx}:")
            for file_path, locations in edit_loc_dict.items():
                print(f"\t\tFile: {file_path}")
                if (
                    locations and locations[0]
                ):  # Check if there are non-empty locations
                    for loc in locations:
                        if loc:  # Skip empty strings
                            print(f"\t\t\t{loc.replace('\n', '\n\t\t\t')}")
                else:
                    print("\t\t\t(no specific location)")

        print("\n" + "=" * 80)
