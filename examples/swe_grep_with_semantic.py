"""
Integration example: SWE-Grep environment with semantic search.

This demonstrates how to integrate semantic search into the existing
SWE-Grep environment for code localization.
"""

import logging

import verifiers as vf
from datasets import load_dataset

import src.rewards as rewards
import src.tools as tools
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.utils.get_instance import get_instance_path

logger = logging.getLogger("swe-grep-oss-semantic")


# Enhanced system prompt with semantic search guidance
SYSTEM_PROMPT_WITH_SEMANTIC = """
You are a helpful assistant that finds files in the codebase that are relevant to the user's query and returns them.
You should not answer the user's query directly. Just find the files and return them.

You have access to THREE search tools:

1. **bash** - Execute bash commands (grep, rg, find, cat, etc.)
   - Best for: Exact keyword matches, file patterns, reading files
   - Example: bash("rg 'class MyClass' -t py")

2. **semantic_search** - Find code by meaning using vector embeddings
   - Best for: Conceptual queries, finding code by description
   - Example: semantic_search("function that calculates F1 score")
   - Returns: File paths and code chunks with similarity scores

3. **result** - Return the final list of relevant file paths
   - ALWAYS use this tool to return your findings

SEARCH STRATEGY:

For most queries, follow this approach:
1. Start with semantic_search to find conceptually relevant files
2. Use bash/grep to verify and explore those files
3. Use bash/cat to read specific files if needed
4. Call result() with the list of relevant file paths

When to use each tool:
- Use semantic_search for: "find code that does X", "functions related to Y"
- Use bash/grep for: "find files containing 'exact_string'"
- Combine both for comprehensive search

ALWAYS use the result tool before responding to the user.
"""


class SWEGrepEnvWithSemantic(vf.StatefulToolEnv):
    """SWE-Grep environment with semantic search integration."""

    def __init__(self, use_semantic_search: bool = True, **kwargs):
        super().__init__(**kwargs)

        # Add standard tools
        self.add_tool(tools.bash, args_to_skip=["cwd"])
        self.add_tool(tools.result)

        # Add semantic search if enabled
        self.use_semantic_search = use_semantic_search
        if use_semantic_search:
            self.add_tool(
                tools.semantic_search, args_to_skip=["repo_path", "rebuild_index"]
            )

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.types.Messages,
        state: vf.types.State,
        **kwargs,
    ) -> dict:
        """Inject repository path into tools."""
        repo_path = get_instance_path(
            {
                "repo": state["info"]["repo"],
                "instance_id": state["info"]["instance_id"],
            }
        )

        if tool_name == "bash":
            updated_tool_args = dict(tool_args)
            updated_tool_args["cwd"] = repo_path
            return updated_tool_args

        if tool_name == "semantic_search":
            updated_tool_args = dict(tool_args)
            updated_tool_args["repo_path"] = str(repo_path)
            # Only rebuild index on first search
            # (Could track this in state if needed)
            updated_tool_args["rebuild_index"] = False
            return updated_tool_args

        return tool_args


def load_environment_with_semantic(use_semantic: bool = True, **kwargs):
    """
    Load environment with optional semantic search.

    Args:
        use_semantic: Whether to enable semantic search tool
        **kwargs: Additional arguments for the environment

    Returns:
        Configured SWEGrepEnvWithSemantic instance
    """
    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset = dataset.map(
        lambda row: {
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
            },
            "prompt": [{"role": "user", "content": row["problem_statement"]}],
            "answer": row["patch"],
        }
    )

    # Define rubric
    rubric = vf.Rubric(
        funcs=[
            rewards.result_tool_check,
            rewards.result_tool_f1,
            rewards.result_tool_precision,
            rewards.result_tool_recall,
        ],
        weights=[2.0, 1.0, 1.0, 1.0],
    )

    # Select system prompt based on semantic search usage
    system_prompt = SYSTEM_PROMPT_WITH_SEMANTIC if use_semantic else SYSTEM_PROMPT

    # Load environment
    return SWEGrepEnvWithSemantic(
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        max_turns=10,
        use_semantic_search=use_semantic,
        **kwargs,
    )


def compare_semantic_vs_baseline():
    """
    Compare performance with and without semantic search.

    This is a utility function to evaluate the benefit of semantic search
    on the SWE-bench benchmark.
    """
    import json
    from datetime import datetime

    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {},
        "with_semantic": {},
    }

    # Test baseline (no semantic search)
    logger.info("Testing baseline (without semantic search)...")
    # Add your evaluation logic here

    # Test with semantic search
    logger.info("Testing with semantic search...")
    # Add your evaluation logic here

    # Save results
    output_file = f"semantic_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    """Example usage."""
    import os

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Example 1: Load environment with semantic search
    print("Loading environment with semantic search...")
    env_with_semantic = load_environment_with_semantic(use_semantic=True)
    print(f"Tools available: {[tool.name for tool in env_with_semantic.tools]}")

    # Example 2: Load environment without semantic search (baseline)
    print("\nLoading baseline environment...")
    env_baseline = load_environment_with_semantic(use_semantic=False)
    print(f"Tools available: {[tool.name for tool in env_baseline.tools]}")

    # Example 3: Compare performance
    if os.getenv("RUN_COMPARISON"):
        print("\nRunning comparison...")
        compare_semantic_vs_baseline()
