#!/usr/bin/env python3
"""
Analyze SWE-bench Lite repository distribution to inform indexing strategy.

This helps answer:
- How many unique repos?
- Which repos appear most frequently?
- What's the optimal frequency threshold for pre-indexing?
"""

import argparse
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Analyze SWE-bench repo distribution")
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (test/train/dev)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=3,
        help="Minimum frequency threshold for pre-indexing recommendation",
    )
    args = parser.parse_args()

    # Load dataset
    from datasets import load_dataset

    print(f"Loading {args.dataset} ({args.split} split)...")
    dataset = load_dataset(args.dataset, split=args.split)

    # Analyze repo distribution
    repos = [instance["repo"] for instance in dataset]
    repo_counts = Counter(repos)

    # Analyze (repo, commit) combinations
    repo_commits = {}
    for instance in dataset:
        key = (instance["repo"], instance["base_commit"])
        if key not in repo_commits:
            repo_commits[key] = 0
        repo_commits[key] += 1

    print("\n" + "=" * 80)
    print("SWE-bench Lite Repository Analysis")
    print("=" * 80)
    print(f"Total instances: {len(dataset)}")
    print(f"Unique repositories: {len(repo_counts)}")
    print(f"Unique (repo, commit) combinations: {len(repo_commits)}")
    print(f"Average instances per repo: {len(dataset) / len(repo_counts):.1f}")
    print()

    print("=" * 80)
    print(f"Top 20 Most Frequent Repositories")
    print("=" * 80)
    cumulative_instances = 0
    for i, (repo, count) in enumerate(repo_counts.most_common(20), 1):
        cumulative_instances += count
        coverage = (cumulative_instances / len(dataset)) * 100
        print(f"{i:2}. {repo:50} {count:3} instances ({coverage:5.1f}% cumulative)")

    print()
    print("=" * 80)
    print("Repository Frequency Distribution")
    print("=" * 80)
    frequency_buckets = {
        "1 instance (singleton)": sum(1 for c in repo_counts.values() if c == 1),
        "2-3 instances": sum(1 for c in repo_counts.values() if 2 <= c <= 3),
        "4-5 instances": sum(1 for c in repo_counts.values() if 4 <= c <= 5),
        "6-10 instances": sum(1 for c in repo_counts.values() if 6 <= c <= 10),
        "11-20 instances": sum(1 for c in repo_counts.values() if 11 <= c <= 20),
        ">20 instances": sum(1 for c in repo_counts.values() if c > 20),
    }

    for bucket, count in frequency_buckets.items():
        print(f"  {bucket:25} {count:3} repos")

    print()
    print("=" * 80)
    print(f"Pre-indexing Strategy Recommendation (min_frequency={args.min_frequency})")
    print("=" * 80)

    # Calculate what percentage of instances would be covered
    repos_to_preindex = [
        (repo, count) for repo, count in repo_counts.items() if count >= args.min_frequency
    ]
    instances_covered = sum(count for _, count in repos_to_preindex)
    coverage_pct = (instances_covered / len(dataset)) * 100

    print(f"Repos to pre-index: {len(repos_to_preindex)} (>= {args.min_frequency} instances)")
    print(f"Instances covered: {instances_covered}/{len(dataset)} ({coverage_pct:.1f}%)")
    print(
        f"Repos for on-demand: {len(repo_counts) - len(repos_to_preindex)} (< {args.min_frequency} instances)"
    )
    print()

    # Estimate space savings
    print("Space Estimation:")
    print(f"  If ALL repos indexed: ~{len(repo_commits)} indices")
    print(
        f"  If frequency >= {args.min_frequency}: ~{len(repos_to_preindex)} indices ({len(repos_to_preindex)/len(repo_commits)*100:.0f}% of total)"
    )
    print(
        f"  Space saved: ~{len(repo_commits) - len(repos_to_preindex)} indices ({(len(repo_commits) - len(repos_to_preindex))/len(repo_commits)*100:.0f}%)"
    )

    # Show recommended repos for pre-indexing
    print()
    print("=" * 80)
    print(f"Recommended Repos for Pre-indexing (>= {args.min_frequency} instances)")
    print("=" * 80)
    for repo, count in sorted(repos_to_preindex, key=lambda x: x[1], reverse=True):
        print(f"  {repo:50} {count:3} instances")

    # Save to file
    output_file = Path("recommended_repos_for_preindexing.txt")
    with open(output_file, "w") as f:
        for repo, count in sorted(repos_to_preindex, key=lambda x: x[1], reverse=True):
            f.write(f"{repo}\n")
    print()
    print(f"✓ Saved recommended repos to: {output_file}")


if __name__ == "__main__":
    main()
