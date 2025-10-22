"""Parse git diff patches to extract file paths and line ranges."""

import re


def parse_patch(patch: str) -> dict:
    """
    Parse a git diff patch and extract file paths with their line ranges.

    Args:
        patch: Git diff patch string

    Returns:
        Dictionary mapping file paths to their modified line ranges:
        {
            "file_path": {
                "old_start": int,
                "old_count": int,
                "new_start": int,
                "new_count": int,
                "hunks": [
                    {
                        "old_start": int,
                        "old_count": int,
                        "new_start": int,
                        "new_count": int
                    },
                    ...
                ]
            },
            ...
        }
    """
    result = {}

    # Split patch into individual file diffs
    file_diffs = re.split(r"^diff --git ", patch, flags=re.MULTILINE)

    for file_diff in file_diffs:
        if not file_diff.strip():
            continue

        # Extract file path from the diff header
        # Format: a/path/to/file b/path/to/file
        file_match = re.search(
            r"a/(.*?) b/(.*?)$", file_diff, flags=re.MULTILINE
        )
        if not file_match:
            continue

        file_path = file_match.group(2)  # Use the 'b/' path (new file)

        # Find all hunks in this file diff
        # Hunk header format: @@ -old_start,old_count +new_start,new_count @@
        hunk_pattern = r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
        hunks = []

        for match in re.finditer(hunk_pattern, file_diff):
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1

            hunks.append(
                {
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                }
            )

        if hunks:
            # Calculate overall ranges
            old_start = min(h["old_start"] for h in hunks)
            old_end = max(h["old_start"] + h["old_count"] - 1 for h in hunks)
            new_start = min(h["new_start"] for h in hunks)
            new_end = max(h["new_start"] + h["new_count"] - 1 for h in hunks)

            result[file_path] = {
                "old_start": old_start,
                "old_count": old_end - old_start + 1,
                "new_start": new_start,
                "new_count": new_end - new_start + 1,
                "hunks": hunks,
            }

    return result


def add_patch_info(example):
    """
    Dataset transformation function to add parsed patch info.

    Args:
        example: Dataset example with 'patch' field

    Returns:
        Example with added 'patch_info' field
    """
    example["patch_info"] = parse_patch(example["patch"])
    return example


if __name__ == "__main__":
    from datasets import load_dataset

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Get first instance
    instance = dataset[0]

    print("\n" + "=" * 80)
    print("Instance ID:", instance["instance_id"])
    print("=" * 80)

    # Show original patch
    print("\nOriginal patch:")
    print("-" * 80)
    print(instance["patch"])

    # Parse the patch
    patch_info = parse_patch(instance["patch"])

    print("\n" + "=" * 80)
    print("Full patch_info dict:")
    print("=" * 80)
    import json

    print(json.dumps(patch_info, indent=2))
