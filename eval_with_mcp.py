#!/usr/bin/env python3
"""
Evaluate SWE-bench using agent-sdk with MCP integration.

This uses the proper MCP server approach for semantic search,
integrating with the agent-sdk framework.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, AgentContext, Skill
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool

import src.rewards as rewards
from src.utils.parse_patch import extract_modified_files


def clone_if_needed(repo_name: str, commit_id: str, instance_id: str, base_dir: Path):
    """Clone repo if it doesn't exist."""
    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = base_dir / instance_dir_name

    if instance_path.exists():
        return instance_path

    print(f"Cloning {instance_id}...")
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo_name}.git", str(instance_path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(instance_path), "checkout", commit_id],
        check=True,
        capture_output=True,
    )

    return instance_path


def extract_file_list_from_conversation(conversation: Conversation) -> list[str]:
    """Extract file list from agent's final response or tool calls."""
    # Try to find files mentioned in conversation
    files = []

    # Look through conversation events for file mentions
    for event in conversation.state.events:
        # Check if there are any tool calls that might contain file paths
        if hasattr(event, 'content'):
            content_str = str(event.content)
            # Look for common file patterns
            import re
            # Match Python files, for example
            matches = re.findall(r'[\w/\-\.]+\.py', content_str)
            files.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


async def evaluate_instance(
    instance: dict,
    llm: LLM,
    use_semantic: bool,
    repos_dir: Path,
) -> dict:
    """Evaluate a single instance."""

    # Clone repo if needed
    repo_path = clone_if_needed(
        instance["info"]["repo"],
        instance["info"]["base_commit"],
        instance["info"]["instance_id"],
        repos_dir,
    )

    # Create agent with MCP if semantic search enabled
    if use_semantic:
        # Load semantic search skill
        skill = Skill.from_file(".openhands/skills/semantic-search.md")

        # Configure MCP server
        mcp_config = {
            "mcpServers": {
                "semantic-code-search": {
                    "command": "uv",
                    "args": ["run", "python", "src/mcp_server/semantic_search_server.py"],
                    "env": {}
                }
            }
        }

        context = AgentContext(skills=[skill])
        agent = Agent(llm=llm, agent_context=context, mcp_config=mcp_config)
    else:
        # No MCP, just basic tools
        agent = Agent(llm=llm)

    # Create conversation
    conversation = Conversation(
        agent=agent,
        workspace=str(repo_path),
        max_iterations=8,
    )

    # Build task prompt
    problem_statement = instance["prompt"][0]["content"]

    task_prompt = f"""You are helping to localize files in a codebase for a bug fix.

Repository: {instance["info"]["repo"]}

Problem:
{problem_statement}

Your task:
1. Read the problem description carefully
2. Search the codebase to find ALL files that need to be modified to fix this issue
3. Return ONLY the list of file paths (relative to repo root)

{'You have access to semantic_search for finding code by meaning.' if use_semantic else 'Use grep, rg, or find to search the codebase.'}

Output the final list of files, one per line, with no additional text.
"""

    # Run agent
    try:
        conversation.send_message(task_prompt)
        await conversation.run()

        # Extract file list from conversation
        predicted_files = extract_file_list_from_conversation(conversation)

        # Get ground truth files from patch
        patch = instance["answer"]
        gold_files = extract_modified_files(patch)

        # Calculate metrics
        if not predicted_files:
            return {
                "instance_id": instance["info"]["instance_id"],
                "predicted_files": [],
                "gold_files": gold_files,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "error": "No files predicted",
            }

        # Calculate set-based metrics
        pred_set = set(predicted_files)
        gold_set = set(gold_files)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "instance_id": instance["info"]["instance_id"],
            "predicted_files": predicted_files,
            "gold_files": gold_files,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    except Exception as e:
        return {
            "instance_id": instance["info"]["instance_id"],
            "error": str(e),
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }


async def evaluate(
    use_semantic: bool = False,
    num_instances: int = 50,
    model: str = "deepseek-ai/deepseek-coder-33b-instruct",
    base_url: str = "http://localhost:8000",
    repos_dir: str = "./swebench_repos",
    clone_on_fly: bool = False,
):
    """
    Evaluate using agent-sdk with MCP.

    Args:
        use_semantic: Use semantic search via MCP
        num_instances: Number of instances to eval
        model: Model name
        base_url: vLLM server URL
        repos_dir: Where repos are/will be stored
        clone_on_fly: Clone repos during eval (vs pre-cloned)
    """
    # Setup repos directory
    repos_path = Path(repos_dir)
    repos_path.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print("Loading SWE-bench...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset = dataset.select(range(num_instances))

    # Format for evaluation
    dataset = dataset.map(
        lambda row: {
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
                "base_commit": row["base_commit"],
            },
            "prompt": [{"role": "user", "content": row["problem_statement"]}],
            "answer": row["patch"],
        }
    )

    # Pre-clone if requested
    if not clone_on_fly:
        print(f"\nPre-cloning {num_instances} repositories...")
        for instance in tqdm(dataset, desc="Cloning"):
            clone_if_needed(
                instance["info"]["repo"],
                instance["info"]["base_commit"],
                instance["info"]["instance_id"],
                repos_path,
            )
        print("✓ All repos cloned\n")

    # Configure LLM
    llm = LLM(
        model=model,
        base_url=f"{base_url}/v1",
        api_key=SecretStr("EMPTY"),
        temperature=0.0,
        max_output_tokens=4096,
    )

    print(f"{'='*80}")
    print(f"Evaluation Configuration (Agent-SDK + MCP)")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Semantic Search (MCP): {'✅ Enabled' if use_semantic else '❌ Disabled'}")
    print(f"Instances: {num_instances}")
    print(f"Repos Directory: {repos_path}")
    print(f"Clone Strategy: {'On-the-fly' if clone_on_fly else 'Pre-cloned'}")
    print(f"{'='*80}\n")

    # Evaluate
    results = []
    for i, instance in enumerate(tqdm(dataset, desc="Evaluating")):
        result = await evaluate_instance(
            instance,
            llm,
            use_semantic,
            repos_path,
        )
        results.append(result)

        # Progress update
        if (i + 1) % 10 == 0:
            successful = [r for r in results if r.get("f1", 0) > 0]
            if successful:
                avg_f1 = sum(r["f1"] for r in successful) / len(successful)
                print(f"\nProgress: {i+1}/{num_instances}, Avg F1: {avg_f1:.3f}")

    # Summary
    successful = [r for r in results if "f1" in r and "error" not in r]

    print(f"\n{'='*80}")
    print(f"Results")
    print(f"{'='*80}")
    print(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        print(f"Average F1:        {sum(r['f1'] for r in successful) / len(successful):.3f}")
        print(f"Average Precision: {sum(r['precision'] for r in successful) / len(successful):.3f}")
        print(f"Average Recall:    {sum(r['recall'] for r in successful) / len(successful):.3f}")
    print(f"{'='*80}\n")

    # Save
    suffix = "mcp_semantic" if use_semantic else "mcp_baseline"
    output_file = f"eval_results_{suffix}.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"✓ Saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on SWE-bench with agent-sdk MCP")
    parser.add_argument("--semantic", action="store_true", help="Use semantic search via MCP")
    parser.add_argument("--num-instances", type=int, default=50)
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-33b-instruct")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--repos-dir", default="./swebench_repos")
    parser.add_argument("--clone-on-fly", action="store_true", help="Clone during eval")
    args = parser.parse_args()

    asyncio.run(evaluate(
        use_semantic=args.semantic,
        num_instances=args.num_instances,
        model=args.model,
        base_url=args.base_url,
        repos_dir=args.repos_dir,
        clone_on_fly=args.clone_on_fly,
    ))
