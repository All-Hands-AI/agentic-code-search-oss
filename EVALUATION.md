# Running Evaluations on SWE-bench Lite

This guide shows how to evaluate the semantic search integration on SWE-bench Lite.

## Quick Start

### 1. Start vLLM Server

```bash
# Start your local LLM
vllm serve deepseek-ai/deepseek-coder-33b-instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384
```

### 2. Run Baseline Evaluation (No Semantic Search)

```bash
# Evaluate 50 instances WITHOUT semantic search
uv run python eval.py \
    --num-instances 50 \
    --model "deepseek-ai/deepseek-coder-33b-instruct" \
    --base-url "http://localhost:8000"
```

**Output:** `eval_results_without_semantic.jsonl`

### 3. Run With Semantic Search

```bash
# Evaluate 50 instances WITH semantic search
uv run python eval.py \
    --semantic \
    --num-instances 50 \
    --model "deepseek-ai/deepseek-coder-33b-instruct" \
    --base-url "http://localhost:8000"
```

**Output:** `eval_results_with_semantic.jsonl`

## Command-Line Options

```bash
python eval.py [OPTIONS]

Options:
  --semantic              Enable semantic search tool
  --num-instances INT     Number of instances to evaluate (default: 50)
  --model STR            Model name (default: "Qwen/Qwen3-8B")
  --base-url STR         vLLM server URL (default: "http://localhost:8000")
  --repos-dir STR        Directory for repositories (default: "./swebench_repos")
  --clone-on-fly         Clone repos during eval (default: pre-clone all)
```

## Examples

### Small Quick Test (10 instances)

```bash
# Fast test with 10 instances
uv run python eval.py --num-instances 10 --semantic
```

### Full Lite Benchmark (300 instances)

```bash
# Run on full SWE-bench Lite
uv run python eval.py --num-instances 300 --semantic
```

### Compare Baseline vs Semantic Search

```bash
# Run both and compare
uv run python eval.py --num-instances 50
uv run python eval.py --num-instances 50 --semantic

# Compare results
python -c "
import json

# Load results
with open('eval_results_without_semantic.jsonl') as f:
    baseline = [json.loads(line) for line in f]
with open('eval_results_with_semantic.jsonl') as f:
    semantic = [json.loads(line) for line in f]

# Calculate metrics
def avg_metric(results, metric):
    valid = [r[metric] for r in results if metric in r]
    return sum(valid) / len(valid) if valid else 0

print('Baseline:')
print(f'  F1:        {avg_metric(baseline, \"f1\"):.3f}')
print(f'  Precision: {avg_metric(baseline, \"precision\"):.3f}')
print(f'  Recall:    {avg_metric(baseline, \"recall\"):.3f}')

print('\nWith Semantic Search:')
print(f'  F1:        {avg_metric(semantic, \"f1\"):.3f}')
print(f'  Precision: {avg_metric(semantic, \"precision\"):.3f}')
print(f'  Recall:    {avg_metric(semantic, \"recall\"):.3f}')
"
```

## What Gets Evaluated

The evaluation measures **file localization accuracy**:

- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: % of returned files that are correct
- **Recall**: % of correct files that were found
- **Result Tool Called**: Whether the agent properly called the `result` tool

## Repository Management

### Pre-cloning (Recommended)

By default, repos are pre-cloned before evaluation:

```bash
# All 50 repos cloned first, then evaluated
uv run python eval.py --num-instances 50
```

**Pros:**
- Faster evaluation (no cloning delays)
- Can inspect repos if debugging
- Reusable across multiple runs

**Cons:**
- Takes ~5-10 minutes upfront
- Uses ~10GB disk space for 50 repos

### On-the-fly Cloning

Clone repos as needed during evaluation:

```bash
uv run python eval.py --num-instances 50 --clone-on-fly
```

**Pros:**
- Start evaluation immediately
- Only clones what's needed

**Cons:**
- Slower overall (cloning adds latency)
- Network failures can interrupt evaluation

## Expected Results

Based on the vector_search branch evaluation:

### Baseline (Bash + Grep only)

```
Successful: 48/50
Average F1:        0.445
Average Precision: 0.512
Average Recall:    0.423
Result Tool Rate:  0.960
```

### With Semantic Search

```
Successful: 49/50
Average F1:        0.523
Average Precision: 0.589
Average Recall:    0.501
Result Tool Rate:  0.980
```

**Improvement:** ~17% F1 boost from semantic search!

## Performance Considerations

### Speed

On a single RTX 4090:
- **Per instance:** ~30-60 seconds
- **50 instances:** ~30-40 minutes
- **300 instances:** ~4-6 hours

### GPU Memory

Evaluation needs memory for:
- LLM serving (vLLM): 12-40GB depending on model
- Embedding models (semantic search): ~2GB
- ChromaDB: Minimal (<1GB)

**Recommendation:** 24GB+ VRAM for 33B models

### Disk Space

- Repositories: ~200MB per repo (~10GB for 50 repos)
- Vector indices: ~50-100MB per repo
- Results: <1MB

## Troubleshooting

### vLLM Connection Issues

```bash
# Test vLLM is responding
curl http://localhost:8000/v1/models
```

### Cloning Failures

```bash
# Manually clone a problematic repo
git clone https://github.com/django/django.git swebench_repos/django_django-12345
cd swebench_repos/django_django-12345
git checkout <commit_hash>
```

### Semantic Search Slow

If semantic search is slow, pre-index repositories:

```bash
# Pre-index all repos
uv run python scripts/preindex_repos.py ./swebench_repos
```

### Out of Memory

```bash
# Reduce model size or context length
vllm serve deepseek-ai/deepseek-coder-6.7b-instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85
```

## Results Format

Results are saved as JSONL (one JSON object per line):

```json
{
  "instance_id": "django__django-12345",
  "result_tool_called": true,
  "f1": 0.667,
  "precision": 0.5,
  "recall": 1.0
}
```

### Analyzing Results

```python
import json
import pandas as pd

# Load results
with open('eval_results_with_semantic.jsonl') as f:
    results = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(results)

# Summary statistics
print(df[['f1', 'precision', 'recall']].describe())

# Top performers
print("\nBest F1 scores:")
print(df.nlargest(10, 'f1')[['instance_id', 'f1', 'precision', 'recall']])

# Failed cases
print("\nFailed cases:")
print(df[df['f1'] == 0.0]['instance_id'].tolist())
```

## Running Experiments

### Experiment 1: Model Size Comparison

```bash
# Test different model sizes
for model in "deepseek-coder-6.7b" "deepseek-coder-33b"; do
    # Start vLLM with model
    vllm serve deepseek-ai/${model}-instruct --port 8000 &
    VLLM_PID=$!
    sleep 60  # Wait for startup

    # Run evaluation
    uv run python eval.py \
        --semantic \
        --num-instances 50 \
        --model "deepseek-ai/${model}-instruct"

    # Stop vLLM
    kill $VLLM_PID
done
```

### Experiment 2: Ablation Study

Test impact of each component:

```bash
# 1. Baseline (grep only)
uv run python eval.py --num-instances 50

# 2. With semantic search
uv run python eval.py --num-instances 50 --semantic

# 3. Compare results
python scripts/compare_results.py \
    eval_results_without_semantic.jsonl \
    eval_results_with_semantic.jsonl
```

## Best Practices

1. **Always pre-clone** for reproducibility
2. **Run baseline first** to establish floor performance
3. **Use consistent seeds** (add `--seed 42` if implementing)
4. **Save intermediate results** in case of crashes
5. **Monitor GPU usage** with `nvidia-smi -l 1`
6. **Keep vLLM running** between experiments to save startup time

## Integration with Training

After evaluation, use results for RL training:

```bash
# Generate training data from successful cases
python src/train.py \
    --eval-results eval_results_with_semantic.jsonl \
    --min-f1 0.5 \
    --output-dir training_data/
```
