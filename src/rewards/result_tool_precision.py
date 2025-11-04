import verifiers as vf

from src.utils.result_tool_metrics import calculate_precision, get_file_sets


def result_tool_precision(
    prompt, completion: vf.types.Messages, answer, state, task, info
) -> float:
    """
    Calculate file-level precision.

    Precision = |result_files ∩ patch_files| / |result_files|

    Measures: Of the files the agent identified, what percentage are correct?

    Args:
        answer: Should contain the patch string
    """
    result_files, patch_files = get_file_sets(completion, answer)

    if result_files is None or patch_files is None:
        return 0.0

    return calculate_precision(result_files, patch_files)
