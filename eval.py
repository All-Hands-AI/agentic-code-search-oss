#!/usr/bin/env python3
"""
Evaluate with YOUR code structure (verifiers + optional semantic search).
Uses pre-cloned repos OR clones on-the-fly.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from openai import AsyncOpenAI
import verifiers as vf
from pydantic import SecretStr

# Your code
from src.tools import bash, result
from src.tools.semantic_search import semantic_search  # Direct import
import src.rewards as rewards
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.utils.get_instance import get_instance_path


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


class EvalEnvironment(vf.StatefulToolEnv):
    """Environment for evaluation - with or without semantic search."""
    
    def __init__(self, use_semantic: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        # Add bash and result (always)
        self.add_tool(bash, args_to_skip=["cwd"])
        self.add_tool(result)
        
        # Add semantic search (optional)
        if use_semantic:
            self.add_tool(semantic_search, args_to_skip=["repo_path", "rebuild_index"])
        
        self.use_semantic = use_semantic
    
    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        """Inject repo path into tools."""
        repo_path = get_instance_path({
            "repo": state["info"]["repo"],
            "instance_id": state["info"]["instance_id"],
        })
        
        # Clone if needed
        if not repo_path.exists():
            clone_if_needed(
                state["info"]["repo"],
                state["info"].get("base_commit", "main"),
                state["info"]["instance_id"],
                repo_path.parent,
            )
        
        updated_args = dict(tool_args)
        
        if tool_name == "bash":
            updated_args["cwd"] = str(repo_path)
        elif tool_name == "semantic_search":
            updated_args["repo_path"] = str(repo_path)
            updated_args["rebuild_index"] = False  # Use cached index
        
        return updated_args


async def evaluate(
    use_semantic: bool = False,
    num_instances: int = 50,
    model: str = "Qwen/Qwen3-4B",
    base_url: str = "http://localhost:8000",
    repos_dir: str = "./swebench_repos",
    clone_on_fly: bool = False,
):
    """
    Evaluate using YOUR verifiers setup.
    
    Args:
        use_semantic: Use semantic search tool
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
    
    # Format for verifiers
    dataset = dataset.map(
        lambda row: {
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
                "base_commit": row["base_commit"],  # Important!
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
    
    # Define rewards
    rubric = vf.Rubric(
        funcs=[
            rewards.result_tool_check,
            rewards.result_tool_f1,
            rewards.result_tool_precision,
            rewards.result_tool_recall,
        ],
        weights=[1.0, 1.0, 1.0, 1.0],
    )
    
    # Create environment
    env = EvalEnvironment(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        use_semantic=use_semantic,
        max_turns=8,
    )
    
    print(f"{'='*80}")
    print(f"Evaluation Configuration")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Semantic Search: {'✅ Enabled' if use_semantic else '❌ Disabled'}")
    print(f"Instances: {num_instances}")
    print(f"Repos Directory: {repos_path}")
    print(f"Clone Strategy: {'On-the-fly' if clone_on_fly else 'Pre-cloned'}")
    print(f"{'='*80}\n")
    
    # Setup OpenAI client
    client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="dummy"
    )
    
    # Evaluate
    results = []
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Generate
            output = await env.a_generate(
                inputs=vf.types.GenerateInputs(
                    prompt=[example["prompt"]],
                    answer=[example["answer"]],
                    info=[example["info"]],
                ),
                client=client,
                model=model,
                sampling_args={
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "tool_choice": "auto",  
                },
            )
            
            # Get metrics
            prompt = example["prompt"]
            completion = output.completion[0]
            answer = example["answer"]
            state = output.state[0]
            
            result_check = rewards.result_tool_check(
                prompt, completion, answer, state, None, None
            )
            f1 = rewards.result_tool_f1(
                prompt, completion, answer, state, None, None
            )
            precision = rewards.result_tool_precision(
                prompt, completion, answer, state, None, None
            )
            recall = rewards.result_tool_recall(
                prompt, completion, answer, state, None, None
            )
            
            result = {
                "instance_id": example["info"]["instance_id"],
                "result_tool_called": result_check == 1.0,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
            
            results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0:
                successful = [r for r in results if r.get("f1", 0) > 0]
                if successful:
                    avg_f1 = sum(r["f1"] for r in successful) / len(successful)
                    print(f"\nProgress: {i+1}/{num_instances}, Avg F1: {avg_f1:.3f}")
        
        except Exception as e:
            print(f"\n❌ Error on {example['info']['instance_id']}: {e}")
            results.append({
                "instance_id": example["info"]["instance_id"],
                "error": str(e),
            })
    
    # Summary
    successful = [r for r in results if "f1" in r]
    
    print(f"\n{'='*80}")
    print(f"Results")
    print(f"{'='*80}")
    print(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        print(f"Average F1:        {sum(r['f1'] for r in successful) / len(successful):.3f}")
        print(f"Average Precision: {sum(r['precision'] for r in successful) / len(successful):.3f}")
        print(f"Average Recall:    {sum(r['recall'] for r in successful) / len(successful):.3f}")
        print(f"Result Tool Rate:  {sum(r['result_tool_called'] for r in successful) / len(successful):.3f}")
    print(f"{'='*80}\n")
    
    # Save
    suffix = "with_semantic" if use_semantic else "without_semantic"
    output_file = f"eval_results_{suffix}.jsonl"
    
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"✓ Saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate on SWE-bench")
    parser.add_argument("--semantic", action="store_true", help="Use semantic search")
    parser.add_argument("--num-instances", type=int, default=50)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
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

    