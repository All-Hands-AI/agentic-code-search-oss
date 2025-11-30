#!/usr/bin/env python3
"""
Simple SWE-bench Lite evaluation using agent-sdk with semantic search.

This evaluates file localization on SWE-bench Lite using the vector search
MCP integration.
"""

import os
import asyncio
import json
import re
import subprocess
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, AgentContext
from openhands.sdk.context.skills import Skill


def clone_repo(repo_name: str, commit_id: str, instance_id: str, base_dir: Path) -> Path:
    """Clone repository if it doesn't exist."""
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


def extract_files_from_patch(patch: str) -> list[str]:
    """Extract modified files from a git patch."""
    files = []
    for line in patch.split('\n'):
        if line.startswith('diff --git'):
            # Parse: diff --git a/file.py b/file.py
            match = re.search(r'b/(.+)$', line)
            if match:
                files.append(match.group(1))
    return files


def extract_files_from_response(response: str) -> list[str]:
    """Extract file paths from agent response - handle multiple formats."""
    import re
    
    files = []
    
    # Pattern 1: Clean paths (one per line)
    # src/file.py
    pattern1 = r'^[\w/\-\.]+\.(py|js|java|cpp|c|go|rb|ts|jsx|tsx)$'
    
    # Pattern 2: Markdown list with paths
    # - path/to/file.py (description)
    pattern2 = r'[-*]\s+([\w/\-\.]+\.(py|js|java|cpp|c|go|rb|ts|jsx|tsx))'
    
    # Pattern 3: Backtick wrapped
    # `path/to/file.py`
    pattern3 = r'`([\w/\-\.]+\.(py|js|java|cpp|c|go|rb|ts|jsx|tsx))`'
    
    # Pattern 4: Just paths with word boundaries
    pattern4 = r'\b([\w/\-]+\.(py|js|java|cpp|c|go|rb|ts|jsx|tsx))\b'
    
    for line in response.split('\n'):
        line = line.strip()
        
        # Try pattern 1 (clean paths)
        if re.match(pattern1, line):
            files.append(line)
            continue
        
        # Try pattern 2 (markdown list)
        match = re.search(pattern2, line)
        if match:
            files.append(match.group(1))
            continue
        
        # Try pattern 3 (backticks)
        match = re.search(pattern3, line)
        if match:
            files.append(match.group(1))
            continue
    
    # If no matches yet, try pattern 4 on full text (more lenient)
    if not files:
        matches = re.findall(pattern4, response)
        files = [m[0] for m in matches]
    
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        # Clean up
        f = f.strip('`"\'.,;: ')
        # Remove common prefixes
        f = f.lstrip('./')
        
        if f and f not in seen and not f.startswith('http'):
            seen.add(f)
            unique_files.append(f)
    
    return unique_files
def calculate_metrics(predicted: list[str], gold: list[str]) -> dict:
    """Calculate precision, recall, and F1."""
    pred_set = set(predicted)
    gold_set = set(gold)

    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


async def evaluate_instance(
    instance: dict,
    llm: LLM,
    use_semantic: bool,
    repos_dir: Path,
) -> dict:
    """Evaluate a single instance."""

    # Clone repo
    repo_path = clone_repo(
        instance["repo"],
        instance["base_commit"],
        instance["instance_id"],
        repos_dir,
    )

    # Create agent
    if use_semantic:
        skill = Skill.load(".openhands/skills/semantic-search.md")
        context = AgentContext(skills=[skill])

        # Configure MCP server with absolute path to wrapper
        from pathlib import Path as P
        script_dir = P(__file__).parent.resolve()
        wrapper_path = script_dir / "run_mcp_server.sh"

        mcp_config = {
            "mcpServers": {
                "semantic-code-search": {
                    "command": "bash",
                    "args": [str(wrapper_path)],
                    "env": {
                        "WORKSPACE_PATH": str(repo_path)
                    }
                }
            }
        }

        agent = Agent(llm=llm, agent_context=context, mcp_config=mcp_config)
    else:
        agent = Agent(llm=llm)

    # Create conversation
    conversation = Conversation(
        agent=agent,
        workspace=str(repo_path),
        max_iteration_per_run=10,
    )

    # Build prompt
    system_hint = """
You are helping to localize files that need to be modified to fix a bug.

Your task:
1. Read the problem description
2. Search the codebase to find ALL files that need modification
3. List the file paths (relative to repo root), one per line

""" + ("You have semantic_search tool and bash terminal - use it to find code by meaning.\n" if use_semantic else "Use grep/find to search the codebase.\n")

    task = f"{system_hint}\nProblem:\n{instance['problem_statement']}\n\nList the files:"

    try:
        conversation.send_message(task)
        await conversation.run()

        # Extract predicted files
        response = conversation.agent_final_response() or ""
        predicted_files = extract_files_from_response(response)

        # Get gold files
        gold_files = extract_files_from_patch(instance["patch"])

        # Calculate metrics
        metrics = calculate_metrics(predicted_files, gold_files)

        return {
            "instance_id": instance["instance_id"],
            "predicted_files": predicted_files,
            "gold_files": gold_files,
            **metrics,
        }

    except Exception as e:
        return {
            "instance_id": instance["instance_id"],
            "error": str(e),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }


async def main(
    num_instances: int = 50,
    use_semantic: bool = False,
    model: str = "claude-sonnet-4-5",
    api_key: str = None,
    base_url: str = None,
    repos_dir: str = "./swebench_repos",
):
    """Run evaluation."""

    # Setup
    repos_path = Path(repos_dir)
    repos_path.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print("Loading SWE-bench Lite...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset = dataset.select(range(num_instances))

    # Configure LLM
    if base_url:
        # Local vLLM (OpenAI-compatible)
        # Prefix model with 'openai/' to tell litellm to use OpenAI-compatible API
        vllm_model = f"openai/{model}" if not model.startswith("openai/") else model
        llm = LLM(
            model=vllm_model,
            base_url=f"{base_url}/v1",
            api_key=SecretStr("dummy"),
            custom_llm_provider="openai",
            temperature=0.0,
        )
    else:
        # Claude API
        llm = LLM(
            model=model,
            api_key=SecretStr(api_key or ""),
            temperature=0.0,
        )

    print(f"\n{'='*80}")
    print(f"SWE-bench Lite Evaluation")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Semantic Search: {'✅ Enabled' if use_semantic else '❌ Disabled'}")
    print(f"Instances: {num_instances}")
    print(f"{'='*80}\n")

    # Evaluate
    results = []
    for instance in tqdm(dataset, desc="Evaluating"):
        result = await evaluate_instance(
            {
                "instance_id": instance["instance_id"],
                "repo": instance["repo"],
                "base_commit": instance["base_commit"],
                "problem_statement": instance["problem_statement"],
                "patch": instance["patch"],
            },
            llm,
            use_semantic,
            repos_path,
        )
        results.append(result)

    # Calculate averages
    successful = [r for r in results if "error" not in r]

    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    print(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        avg_f1 = sum(r["f1"] for r in successful) / len(successful)
        avg_precision = sum(r["precision"] for r in successful) / len(successful)
        avg_recall = sum(r["recall"] for r in successful) / len(successful)
        print(f"Average F1:        {avg_f1:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall:    {avg_recall:.3f}")
    print(f"{'='*80}\n")

    # Save results
    suffix = "semantic" if use_semantic else "baseline"
    output_file = f"eval_results_{suffix}.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"✓ Results saved to: {output_file}\n")

    return results


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Evaluate on SWE-bench Lite")
    parser.add_argument("--num-instances", type=int, default=50, help="Number of instances")
    parser.add_argument("--semantic", action="store_true", help="Use semantic search")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Model name")
    parser.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"), help="API key")
    parser.add_argument("--base-url", help="vLLM base URL (e.g., http://localhost:8000)")
    parser.add_argument("--repos-dir", default="./swebench_repos", help="Repos directory")
    args = parser.parse_args()

    asyncio.run(main(
        num_instances=args.num_instances,
        use_semantic=args.semantic,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        repos_dir=args.repos_dir,
    ))