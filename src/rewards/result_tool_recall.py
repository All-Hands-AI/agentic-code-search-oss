import verifiers as vf

from src.utils.result_tool_metrics import calculate_recall, get_file_sets


def result_tool_recall(
    prompt, completion: vf.types.Messages, answer, state, task, info
) -> float:
    """
    Calculate file-level recall.

    Recall = |result_files ∩ patch_files| / |patch_files|

    Measures: Of all the files in the patch, what percentage did the
    agent identify?

    Args:
        answer: Should contain the patch string
    """
    result_files, patch_files = get_file_sets(completion, answer)

    if result_files is None or patch_files is None:
        return 0.0

    return calculate_recall(result_files, patch_files)
