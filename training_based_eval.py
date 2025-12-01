#!/usr/bin/env python3
"""
Evaluation matching the training setup exactly.
Uses the same agent, prompts, and tool configuration as training.
"""

import os
import sys
import json
import ray
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.code_search_generator import init_and_run
from omegaconf import DictConfig


def evaluate_base_model(
    num_instances: int = 100,
    model: str = "Qwen/Qwen3-4B",
    use_semantic: bool = True,
    data_split: str = "test",
):
    """Evaluate base model using training infrastructure."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Load dataset with preprocessing
    print(f"Loading dataset...")
    if "SWE-Gym" in data_split:
        from datasets import load_from_disk
        dataset = load_from_disk("data/SWE-Gym__SWE-Gym_train")
    else:
        # Load SWE-bench Lite and add target field
        from datasets import load_dataset
        import pandas as pd
        
        # Check if preprocessed version exists
        preprocessed_path = "data/SWE-bench_Lite_processed"
        try:
            dataset = load_from_disk(preprocessed_path)
            print(f"Loaded preprocessed data from {preprocessed_path}")
        except:
            print("Preprocessing SWE-bench Lite...")
            dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            
            # Add target field
            from src.utils.dataset import extract_functions_from_patch
            
            # Convert to pandas for processing
            df = dataset.to_pandas()
            df["target"] = df.apply(
                lambda row: f"{extract_functions_from_patch(row['patch'])}", 
                axis=1
            )
            
            # Convert back to dataset
            from datasets import Dataset
            dataset = Dataset.from_pandas(df)
            
            # Save for next time
            dataset.save_to_disk(preprocessed_path)
            print(f"Saved preprocessed data to {preprocessed_path}")
    
    dataset = dataset.select(range(min(num_instances, len(dataset))))
    print(f"Evaluating {len(dataset)} instances...")
    
    # Create generator config
    generator_cfg = DictConfig({
        "use_semantic_search": use_semantic,
        "base_path": "/home/sanidhyv/agentic-code-search-oss",
    })
    
    results = []
    
    for instance in tqdm(dataset):
        try:
            # Convert dataset format to match training
            instance_dict = {
                "instance_id": instance["instance_id"],
                "repo": instance["repo"],
                "base_commit": instance["base_commit"],
                "problem_statement": instance.get("problem_statement", instance.get("problem", "")),
                "patch": instance["patch"],
            }
            
            # Call the same init_and_run function used in training
            result = ray.get(
                init_and_run.remote(
                    instance=instance_dict,
                    litellm_model_name=model,
                    litellm_base_url={"base_url": "http://localhost:8080"},
                    generator_cfg=generator_cfg,
                    data_source="swebench_lite",
                    sampling_params={},
                    trajectory_id=None,
                    global_step=0,
                    training_phase="eval",
                )
            )
            
            (messages, final_message), (reward, reward_dict), error = result
            
            results.append({
                "instance_id": instance_dict["instance_id"],
                "reward": reward,
                "final_message": final_message,
                **reward_dict,
                "error": error,
                "num_messages": len(messages),
            })
            
            # Print progress
            if len(results) % 10 == 0:
                successful = [r for r in results if r["reward"] > 0]
                avg_reward = sum(r["reward"] for r in results) / len(results)
                print(f"\n[Progress] {len(results)}/{len(dataset)} | Avg Reward: {avg_reward:.3f} | Success Rate: {len(successful)}/{len(results)}")
            
        except Exception as e:
            print(f"\nError on {instance['instance_id']}: {e}")
            results.append({
                "instance_id": instance["instance_id"],
                "reward": 0.0,
                "error": str(e),
            })
    
    # Calculate final statistics
    successful = [r for r in results if r["reward"] > 0]
    avg_reward = sum(r["reward"] for r in results) / len(results)
    
    print(f"\n{'='*80}")
    print(f"Base Model Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Semantic Search: {'✅ Enabled' if use_semantic else '❌ Disabled'}")
    print(f"Total Instances: {len(results)}")
    print(f"Successful (reward > 0): {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Average Reward: {avg_reward:.3f}")
    
    if successful:
        avg_f1 = sum(r.get("file_localization_f1", 0) for r in successful) / len(successful)
        avg_multiturn = sum(r.get("multiturn_reward", 0) for r in successful) / len(successful)
        print(f"Average F1 (all): {sum(r.get('file_localization_f1', 0) for r in results) / len(results):.3f}")
        print(f"Average F1 (successful): {avg_f1:.3f}")
        print(f"Average Multiturn (successful): {avg_multiturn:.3f}")
    
    print(f"{'='*80}\n")
    
    # Save results
    suffix = "semantic" if use_semantic else "baseline"
    output_file = f"eval_base_model_{suffix}.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"✓ Results saved to {output_file}\n")
    
    # Shutdown Ray
    ray.shutdown()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate base model matching training setup")
    parser.add_argument("--num-instances", type=int, default=100, help="Number of instances to evaluate")
    parser.add_argument("--model", default="openai/Qwen/Qwen3-4B", help="Model name")
    parser.add_argument("--semantic", action="store_true", help="Enable semantic search")
    parser.add_argument("--data-split", default="test", help="Dataset split")
    args = parser.parse_args()
    
    evaluate_base_model(
        num_instances=args.num_instances,
        model=args.model,
        use_semantic=args.semantic,
        data_split=args.data_split,
    )