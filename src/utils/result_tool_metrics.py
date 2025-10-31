"""Utility functions for calculating result tool metrics."""

import json

import verifiers as vf

from src.utils.get_result_tool_call import get_result_tool_call
from src.utils.parse_patch import parse_patch


def get_file_sets(
    completion: vf.types.Messages, patch: str
) -> tuple[set[str], set[str]] | tuple[None, None]:
    """
    Extract file sets from completion messages and patch.

    Args:
        completion: The completion messages from the agent
        patch: The ground truth patch string

    Returns:
        Tuple of (result_files, patch_files) or (None, None) if extraction fails
    """
    result_tool_call, success = get_result_tool_call(completion)

    # If no successful result tool call, return None
    if not success or not result_tool_call:
        return None, None

    # Parse the patch to get file paths
    patch_info = parse_patch(patch)
    patch_files = set(patch_info.keys())

    # Get file paths from the result tool call
    try:
        # Parse the arguments from the tool call
        if hasattr(result_tool_call, "function") and hasattr(
            result_tool_call.function, "arguments"
        ):
            args_str = result_tool_call.function.arguments
            args = json.loads(args_str)

            # Get file paths from the result
            result_files = set()
            if "file_paths" in args:
                result_files = set(args["file_paths"])

            if not result_files:
                return None, None

            return result_files, patch_files
    except (json.JSONDecodeError, AttributeError, KeyError):
        return None, None

    return None, None


def calculate_precision(result_files: set[str], patch_files: set[str]) -> float:
    """
    Calculate precision: proportion of predicted files that are correct.

    Precision = |result_files ∩ patch_files| / |result_files|

    Args:
        result_files: Files identified by the agent
        patch_files: Files in the ground truth patch

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not result_files:
        return 0.0

    matching_files = result_files.intersection(patch_files)
    return len(matching_files) / len(result_files)


def calculate_recall(result_files: set[str], patch_files: set[str]) -> float:
    """
    Calculate recall: proportion of ground truth files that were identified.

    Recall = |result_files ∩ patch_files| / |patch_files|

    Args:
        result_files: Files identified by the agent
        patch_files: Files in the ground truth patch

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not patch_files:
        return 0.0

    matching_files = result_files.intersection(patch_files)
    return len(matching_files) / len(patch_files)


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score: harmonic mean of precision and recall.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score between 0.0 and 1.0
    """
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
