#!/usr/bin/env python3
"""Compare evaluation results from baseline vs semantic search."""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_results(filepath: str) -> List[Dict]:
    """Load JSONL results file."""
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate average metrics from results."""
    successful = [r for r in results if "f1" in r]

    if not successful:
        return {
            "count": 0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "result_tool_rate": 0.0,
        }

    return {
        "count": len(successful),
        "f1": sum(r["f1"] for r in successful) / len(successful),
        "precision": sum(r["precision"] for r in successful) / len(successful),
        "recall": sum(r["recall"] for r in successful) / len(successful),
        "result_tool_rate": sum(r.get("result_tool_called", 0) for r in successful) / len(successful),
    }


def find_improvements(baseline: List[Dict], semantic: List[Dict]) -> List[Dict]:
    """Find instances where semantic search improved results."""
    baseline_dict = {r["instance_id"]: r for r in baseline if "f1" in r}
    semantic_dict = {r["instance_id"]: r for r in semantic if "f1" in r}

    improvements = []
    for instance_id in baseline_dict:
        if instance_id in semantic_dict:
            base_f1 = baseline_dict[instance_id]["f1"]
            sem_f1 = semantic_dict[instance_id]["f1"]

            if sem_f1 > base_f1 + 0.05:  # Significant improvement
                improvements.append({
                    "instance_id": instance_id,
                    "baseline_f1": base_f1,
                    "semantic_f1": sem_f1,
                    "improvement": sem_f1 - base_f1,
                })

    return sorted(improvements, key=lambda x: x["improvement"], reverse=True)


def find_regressions(baseline: List[Dict], semantic: List[Dict]) -> List[Dict]:
    """Find instances where semantic search made results worse."""
    baseline_dict = {r["instance_id"]: r for r in baseline if "f1" in r}
    semantic_dict = {r["instance_id"]: r for r in semantic if "f1" in r}

    regressions = []
    for instance_id in baseline_dict:
        if instance_id in semantic_dict:
            base_f1 = baseline_dict[instance_id]["f1"]
            sem_f1 = semantic_dict[instance_id]["f1"]

            if sem_f1 < base_f1 - 0.05:  # Significant regression
                regressions.append({
                    "instance_id": instance_id,
                    "baseline_f1": base_f1,
                    "semantic_f1": sem_f1,
                    "regression": base_f1 - sem_f1,
                })

    return sorted(regressions, key=lambda x: x["regression"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("baseline", help="Baseline results JSONL file")
    parser.add_argument("semantic", help="Semantic search results JSONL file")
    parser.add_argument("--output", help="Save comparison to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown")
    args = parser.parse_args()

    # Load results
    print("Loading results...")
    baseline = load_results(args.baseline)
    semantic = load_results(args.semantic)

    # Calculate metrics
    baseline_metrics = calculate_metrics(baseline)
    semantic_metrics = calculate_metrics(semantic)

    # Print comparison
    print("\n" + "=" * 80)
    print("EVALUATION COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Baseline':<15} {'Semantic':<15} {'Δ':<10} {'%':<10}")
    print("-" * 80)

    metrics = ["f1", "precision", "recall", "result_tool_rate"]
    for metric in metrics:
        base_val = baseline_metrics[metric]
        sem_val = semantic_metrics[metric]
        delta = sem_val - base_val
        pct_change = (delta / base_val * 100) if base_val > 0 else 0

        delta_str = f"{delta:+.3f}" if metric != "result_tool_rate" else f"{delta:+.3f}"
        pct_str = f"{pct_change:+.1f}%" if metric != "result_tool_rate" else f"{pct_change:+.1f}%"

        print(f"{metric:<20} {base_val:<15.3f} {sem_val:<15.3f} {delta_str:<10} {pct_str:<10}")

    print("-" * 80)
    print(f"{'Instances':<20} {baseline_metrics['count']:<15} {semantic_metrics['count']:<15}")
    print("=" * 80)

    # Find improvements and regressions
    if args.verbose:
        improvements = find_improvements(baseline, semantic)
        regressions = find_regressions(baseline, semantic)

        if improvements:
            print(f"\n🎯 TOP IMPROVEMENTS ({len(improvements)} instances):")
            print("-" * 80)
            for i, imp in enumerate(improvements[:10], 1):
                print(f"{i:2}. {imp['instance_id']:<40} "
                      f"{imp['baseline_f1']:.3f} → {imp['semantic_f1']:.3f} "
                      f"({imp['improvement']:+.3f})")

        if regressions:
            print(f"\n⚠️  REGRESSIONS ({len(regressions)} instances):")
            print("-" * 80)
            for i, reg in enumerate(regressions[:10], 1):
                print(f"{i:2}. {reg['instance_id']:<40} "
                      f"{reg['baseline_f1']:.3f} → {reg['semantic_f1']:.3f} "
                      f"({-reg['regression']:+.3f})")

        # Summary
        improved = len(improvements)
        regressed = len(regressions)
        total = len([r for r in baseline if "f1" in r])
        unchanged = total - improved - regressed

        print(f"\n📊 CHANGE DISTRIBUTION:")
        print("-" * 80)
        print(f"Improved:  {improved:3} ({improved/total*100:5.1f}%)")
        print(f"Unchanged: {unchanged:3} ({unchanged/total*100:5.1f}%)")
        print(f"Regressed: {regressed:3} ({regressed/total*100:5.1f}%)")

    # Save detailed comparison if requested
    if args.output:
        comparison = {
            "baseline_file": args.baseline,
            "semantic_file": args.semantic,
            "baseline_metrics": baseline_metrics,
            "semantic_metrics": semantic_metrics,
            "improvements": find_improvements(baseline, semantic),
            "regressions": find_regressions(baseline, semantic),
        }

        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n✓ Detailed comparison saved to: {args.output}")

    # Conclusion
    print("\n" + "=" * 80)
    f1_delta = semantic_metrics["f1"] - baseline_metrics["f1"]
    if f1_delta > 0.05:
        print(f"✅ SEMANTIC SEARCH IMPROVED PERFORMANCE BY {f1_delta:.3f} F1 ({f1_delta/baseline_metrics['f1']*100:.1f}%)")
    elif f1_delta < -0.05:
        print(f"⚠️  SEMANTIC SEARCH DECREASED PERFORMANCE BY {-f1_delta:.3f} F1")
    else:
        print(f"➖ NO SIGNIFICANT DIFFERENCE ({f1_delta:+.3f} F1)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
