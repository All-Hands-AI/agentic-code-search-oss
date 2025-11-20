#!/usr/bin/env python3
"""
Pre-index all SWE-bench repos for fast RL training.

This script indexes repositories before training begins so that
semantic search calls during training are fast (<1 sec).
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.tools.semantic_search import SemanticSearchIndex
from src.utils.get_instance import get_instance_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preindex_repos(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    repos_dir: Path = None,
    split: str = "test",
    force_rebuild: bool = False,
):
    """
    Pre-index all repositories in the dataset.
    
    Args:
        dataset_name: SWE-bench dataset to use
        repos_dir: Directory containing cloned repos
        split: Dataset split (train/test)
        force_rebuild: Whether to rebuild existing indices
    """
    logger.info(f"Loading dataset: {dataset_name} (split={split})")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"✓ Loaded {len(dataset)} instances")
    
    # Get unique repository paths
    unique_repos = {}  # repo_path -> [instance_ids]
    for instance in dataset:
        repo_path = get_instance_path(instance, repos_dir)
        if repo_path.exists():
            if repo_path not in unique_repos:
                unique_repos[repo_path] = []
            unique_repos[repo_path].append(instance["instance_id"])
        else:
            logger.warning(f"Repository not found: {repo_path}")
    
    logger.info(f"Found {len(unique_repos)} unique repository instances")
    logger.info(f"Total storage will be used: ~{len(unique_repos) * 50}MB")
    
    # Index statistics
    stats = {
        "total_repos": len(unique_repos),
        "indexed": 0,
        "skipped": 0,
        "failed": 0,
    }
    
    # Index each repo
    for repo_path in tqdm(unique_repos.keys(), desc="Indexing repos"):
        try:
            # Create index
            index = SemanticSearchIndex(
                collection_name=f"code_index_{repo_path.name}",
                persist_directory=str(repo_path / ".vector_index"),
            )
            
            # Check if already indexed
            index_stats = index.get_stats()
            if index_stats["total_documents"] > 0 and not force_rebuild:
                logger.info(
                    f"✓ {repo_path.name} already indexed "
                    f"({index_stats['total_documents']} docs)"
                )
                stats["skipped"] += 1
                continue
            
            # Index the repository
            if force_rebuild:
                logger.info(f"Rebuilding index for {repo_path.name}...")
                index.clear_index()
            else:
                logger.info(f"Indexing {repo_path.name}...")
            
            result = index.index_code_files(
                str(repo_path),
                file_extensions=[".py"],  # Only Python for code localization
            )
            
            logger.info(
                f"✓ Indexed {result['indexed_files']} files, "
                f"{result['total_chunks']} chunks"
            )
            stats["indexed"] += 1
            
        except Exception as e:
            logger.error(f"✗ Error indexing {repo_path.name}: {e}")
            stats["failed"] += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("Indexing Summary")
    print("=" * 80)
    print(f"Total repositories: {stats['total_repos']}")
    print(f"Successfully indexed: {stats['indexed']}")
    print(f"Already indexed (skipped): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print("=" * 80)
    
    if stats["indexed"] > 0:
        print(f"\n✓ Indexed {stats['indexed']} new repositories!")
        print("Your RL training will now be much faster! 🚀")
    elif stats["skipped"] == stats["total_repos"]:
        print("\n✓ All repositories already indexed!")
        print("Ready for RL training! 🚀")
    
    if stats["failed"] > 0:
        print(f"\n⚠ Warning: {stats['failed']} repositories failed to index")
        print("Check the logs above for details")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-index SWE-bench repositories for RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index training split
  %(prog)s --split train
  
  # Index test split
  %(prog)s --split test
  
  # Rebuild existing indices
  %(prog)s --split train --force-rebuild
  
  # Use custom repos directory
  %(prog)s --repos-dir ~/swebench_repos --split train
        """,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset to use (default: princeton-nlp/SWE-bench_Lite)",
    )
    
    parser.add_argument(
        "--repos-dir",
        type=str,
        default=None,
        help="Directory containing cloned repos (default: ./swebench_repos)",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to index (default: train)",
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild existing indices",
    )
    
    args = parser.parse_args()
    
    # Convert repos_dir to Path if provided
    repos_dir = Path(args.repos_dir) if args.repos_dir else None
    
    # Run indexing
    preindex_repos(
        dataset_name=args.dataset,
        repos_dir=repos_dir,
        split=args.split,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
