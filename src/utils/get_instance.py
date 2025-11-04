import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _default_repos_dir() -> Path:
    """Resolve the default path for the cloned SWE-bench repositories.

    Priority order:
    1) Environment variable SWEBENCH_REPOS_DIR if set
    2) Project root joined with "swebench_repos" (project root
       inferred from this file)
    3) Current working directory joined with "swebench_repos" as a final
       fallback
    """
    env_override = os.getenv("SWEBENCH_REPOS_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()

    # This file lives at <project_root>/src/utils/get_instance.py
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "swebench_repos"
    if candidate.exists() or candidate.parent.exists():
        return candidate

    # Fallback to CWD
    return Path.cwd() / "swebench_repos"


def get_instance_path(instance: dict, output_dir: Path | None = None) -> Path:
    """
    Get the filesystem path for a SWE-bench instance.

    Args:
        instance: Dictionary with 'repo' and 'instance_id' keys
        output_dir: Base directory where instances are cloned

    Returns:
        Path to the instance directory
    """
    if output_dir is None:
        output_dir = _default_repos_dir()

    repo_name = instance["repo"]
    instance_id = instance["instance_id"]
    dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    return output_dir / dir_name


def main():
    parser = argparse.ArgumentParser(
        description="Get filesystem path for SWE-bench instance"
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        help="Instance ID to look up (e.g., astropy__astropy-12907)",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Dataset index to look up (e.g., 0 for first instance)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_default_repos_dir()),
        help=(
            "Base directory where instances are cloned. Defaults to "
            "SWEBENCH_REPOS_DIR if set, else <project_root>/swebench_repos"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset to use",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the path exists and show info",
    )

    args = parser.parse_args()

    if not args.instance_id and args.index is None:
        parser.error("Either --instance-id or --index must be provided")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test")
    print(f"✓ Loaded {len(dataset)} instances\n")

    # Find the instance
    if args.instance_id:
        # Find by instance_id
        instance = None
        for inst in dataset:
            if inst["instance_id"] == args.instance_id:
                instance = inst
                break
        if not instance:
            print(f"✗ Instance ID '{args.instance_id}' not found in dataset")
            return
    else:
        # Get by index
        if args.index < 0 or args.index >= len(dataset):
            print(f"✗ Index {args.index} out of range [0, {len(dataset)-1}]")
            return
        instance = dataset[args.index]

    # Get the path
    output_dir = Path(args.output_dir)
    instance_path = get_instance_path(instance, output_dir)

    # Display info
    print("=" * 80)
    print("Instance Information:")
    print("=" * 80)
    print(f"Instance ID:  {instance['instance_id']}")
    print(f"Repository:   {instance['repo']}")
    print(f"Base Commit:  {instance['base_commit']}")
    print("\nFilesystem Path:")
    print(f"  {instance_path.absolute()}")

    # Check if exists
    if args.check:
        print("\n" + "=" * 80)
        print("Path Check:")
        print("=" * 80)
        if instance_path.exists():
            print("✓ Path exists")
            print("\nDirectory contents (first 10 items):")
            items = sorted(instance_path.iterdir())[:10]
            for item in items:
                item_type = "📁" if item.is_dir() else "📄"
                print(f"  {item_type} {item.name}")
            all_items = list(instance_path.iterdir())
            if len(all_items) > 10:
                extra_count = len(all_items) - 10
                print(f"  ... and {extra_count} more")

            # Count Python files
            py_files = list(instance_path.rglob("*.py"))
            print(f"\nTotal Python files: {len(py_files)}")
        else:
            print("✗ Path does not exist")
            print("\nTo clone this instance, run:")
            cmd = (
                f"  python scripts/clone_repos.py --max-instances 1 "
                f'--dataset "{args.dataset}"'
            )
            print(cmd)


if __name__ == "__main__":
    main()
