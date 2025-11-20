#!/usr/bin/env python3
"""
CLI tool for semantic code search.

Usage:
    python cli_semantic_search.py <repo_path> <query> [options]

Examples:
    python cli_semantic_search.py . "function that calculates F1 score"
    python cli_semantic_search.py ~/projects/myrepo "validation logic" --n-results 5
    python cli_semantic_search.py . "reward function" --rebuild
"""

import argparse
import sys
from pathlib import Path

from src.tools.semantic_search import SemanticSearch


def main():
    parser = argparse.ArgumentParser(
        description="Semantic code search using vector embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s . "function that calculates F1 score"
  %(prog)s ~/projects/myrepo "validation logic" --n-results 5
  %(prog)s . "reward function" --rebuild --files-only

Search Tips:
  - Use descriptive queries: "function that X" instead of just "X"
  - Be specific about what you're looking for
  - Combine with grep for best results
        """,
    )

    parser.add_argument(
        "repo_path", type=str, help="Path to the repository to search"
    )

    parser.add_argument("query", type=str, help="Natural language search query")

    parser.add_argument(
        "-n",
        "--n-results",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from scratch",
    )

    parser.add_argument(
        "--files-only",
        action="store_true",
        help="Show only unique file paths (no content)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics and exit",
    )

    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".py"],
        help="File extensions to index (default: .py)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the index and exit",
    )

    args = parser.parse_args()

    # Resolve repository path
    repo_path = Path(args.repo_path).resolve()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize index
    index = SemanticSearch(
    embedding_model_name=args.model,
    collection_name=f"code_index_{repo_path.name}",
    reranker_model_name="jinaai/jina-reranker-v3",
    )

    # Handle special commands
    if args.clear:
        print(f"Clearing index for {repo_path}...")
        index.clear_index()
        print("✓ Index cleared")
        return

    if args.stats:
        stats = index.get_stats()
        print("=" * 80)
        print("Index Statistics")
        print("=" * 80)
        print(f"Repository: {repo_path}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Index directory: {stats['persist_directory']}")
        print("=" * 80)
        return

    # Check if index needs to be built
    stats = index.get_stats()
    if args.rebuild or stats["total_documents"] == 0:
        print(f"Building index for {repo_path}...")
        print(f"File extensions: {args.extensions}")
        print()

        index_stats = index.index_code_files(
            str(repo_path), file_extensions=args.extensions
        )

        print(f"✓ Indexed {index_stats['indexed_files']} files")
        print(f"✓ Created {index_stats['total_chunks']} chunks")
        print()

    # Perform search
    print(f"Searching for: '{args.query}'")
    print("=" * 80)
    print()

    results = index.search(args.query, n_results=args.n_results)

    if not results:
        print("No results found.")
        return

    if args.files_only:
        # Show only unique files
        unique_files = index.get_unique_files(results)
        print(f"Found {len(unique_files)} unique files:\n")
        for file_path in unique_files:
            print(f"  {file_path}")
    else:
        # Show full results
        print(f"Found {len(results)} relevant code chunks:\n")

        for i, result in enumerate(results, 1):
            similarity = result["similarity_score"]
            file_path = result["file_path"]
            chunk_idx = result["chunk_index"]
            total_chunks = result["metadata"]["total_chunks"]

            print(f"{i}. {file_path} (similarity: {similarity:.3f})")
            print(f"   Chunk {chunk_idx + 1}/{total_chunks}")

            # Show content preview
            content = result["content"]
            lines = content.split("\n")

            # Show first 10 lines
            preview_lines = min(10, len(lines))
            for line in lines[:preview_lines]:
                print(f"   {line}")

            if len(lines) > preview_lines:
                print(f"   ... ({len(lines)} total lines)")

            print()

        # Show unique files summary
        unique_files = index.get_unique_files(results)
        print("=" * 80)
        print(f"Unique files ({len(unique_files)}):")
        for file_path in unique_files:
            print(f"  - {file_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
