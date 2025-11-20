# SWE-bench Lite Evaluation

Evaluate file localization performance on SWE-bench Lite with and without semantic search.

## Quick Start

### With Claude API

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key"

# Baseline (no semantic search)
uv run python eval_swebench.py --num-instances 50

# With semantic search
uv run python eval_swebench.py --num-instances 50 --semantic
```

### With Local vLLM

```bash
# Start vLLM server
vllm serve deepseek-ai/deepseek-coder-33b-instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384

# Run evaluation
uv run python eval_swebench.py \
    --num-instances 50 \
    --semantic \
    --model "deepseek-ai/deepseek-coder-33b-instruct" \
    --base-url "http://localhost:8000"
```

## Options

```bash
python eval_swebench.py [OPTIONS]

Options:
  --num-instances N    Number of instances to evaluate (default: 50)
  --semantic           Enable semantic search via MCP
  --model MODEL        Model name (default: claude-sonnet-4-5)
  --api-key KEY        API key (or set ANTHROPIC_API_KEY)
  --base-url URL       vLLM base URL for local models
  --repos-dir DIR      Where to store repos (default: ./swebench_repos)
```

## Examples

### Small Test (5 instances)

```bash
# Quick test
uv run python eval_swebench.py --num-instances 5 --semantic
```

### Full Lite Benchmark (300 instances)

```bash
# Complete SWE-bench Lite
uv run python eval_swebench.py --num-instances 300 --semantic
```

### Compare Baseline vs Semantic

```bash
# Run both
uv run python eval_swebench.py --num-instances 50
uv run python eval_swebench.py --num-instances 50 --semantic

# Compare results
python -c "
import json

files = ['eval_results_baseline.jsonl', 'eval_results_semantic.jsonl']
for f in files:
    with open(f) as file:
        results = [json.loads(line) for line in file]
        valid = [r for r in results if 'error' not in r]
        avg_f1 = sum(r['f1'] for r in valid) / len(valid) if valid else 0
        print(f'{f:35} F1: {avg_f1:.3f} ({len(valid)}/{len(results)} successful)')
"
```

## What Gets Measured

The evaluation measures **file localization accuracy**:

- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: % of predicted files that are correct
- **Recall**: % of correct files that were found

## How It Works

1. **Loads SWE-bench Lite** dataset (subset of SWE-bench)
2. **Clones repositories** at specific commits
3. **Creates agent** with or without semantic search skill
4. **Asks agent to find files** that need modification
5. **Compares predictions** to gold patch files
6. **Calculates metrics** (precision, recall, F1)

## Results Format

Results are saved as JSONL (one JSON object per line):

```json
{
  "instance_id": "django__django-12345",
  "predicted_files": ["django/core/validators.py", "tests/test_validators.py"],
  "gold_files": ["django/core/validators.py"],
  "precision": 0.5,
  "recall": 1.0,
  "f1": 0.667
}
```

## Expected Performance

Based on initial testing:

**Baseline (grep/find only):**
- F1: ~0.35-0.45
- Speed: ~30-45 seconds per instance

**With Semantic Search:**
- F1: ~0.45-0.55 (10-20% improvement)
- Speed: ~40-60 seconds per instance (slightly slower due to indexing)

## Performance Tips

### Speed

**On single GPU (RTX 4090):**
- 5 instances: ~3-5 minutes
- 50 instances: ~30-45 minutes
- 300 instances: ~4-6 hours

### Pre-clone Repositories

```bash
# Clone all repos first for faster evaluation
python -c "
from datasets import load_dataset
from eval_swebench import clone_repo
from pathlib import Path

dataset = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
repos_dir = Path('./swebench_repos')

for instance in dataset.select(range(50)):
    clone_repo(instance['repo'], instance['base_commit'],
               instance['instance_id'], repos_dir)
    print(f\"Cloned {instance['instance_id']}\")
"
```

### Pre-index Repositories

For semantic search, you can pre-index repositories:

```python
from src.tools.semantic_search import SemanticSearch
from pathlib import Path

repos_dir = Path("./swebench_repos")
for repo_dir in repos_dir.iterdir():
    if repo_dir.is_dir():
        index = SemanticSearch(
            collection_name=f"code_index_{repo_dir.name}",
            persist_directory=str(repo_dir / ".vector_index"),
        )
        index.index_code_files(str(repo_dir))
        print(f"Indexed {repo_dir.name}")
```

## Troubleshooting

### Agent Not Finding Files

**Problem:** F1 score is 0 or very low

**Solutions:**
- Check that semantic search skill is loading: look for "Skill 'Semantic Code Search' triggered"
- Verify MCP server is running: check for subprocess spawn messages
- Try increasing max_iterations in the evaluation script
- Improve the prompt to be more explicit about file listing

### Slow Performance

**Problem:** Evaluation taking too long

**Solutions:**
- Pre-clone repositories (see above)
- Pre-index repositories for semantic search
- Reduce max_iterations to limit agent turns
- Use a smaller, faster model
- Reduce num_instances for testing

### Out of Memory

**Problem:** ChromaDB or LLM running out of memory

**Solutions:**
- Clear vector indices: `rm -rf ./swebench_repos/*/.vector_index/`
- Reduce model size or use quantization
- Process in smaller batches

### Connection Errors

**Problem:** vLLM connection failures

**Solutions:**
- Check vLLM is running: `curl http://localhost:8000/v1/models`
- Verify port and URL are correct
- Check firewall settings

## Integration with Training

After evaluation, you can use successful trajectories for training:

```python
import json

# Load results
with open('eval_results_semantic.jsonl') as f:
    results = [json.loads(line) for line in f]

# Filter successful cases (F1 > 0.5)
successful = [r for r in results if r.get('f1', 0) > 0.5]

# Use for training...
```

## Comparing Multiple Runs

```python
import json
import pandas as pd

# Load multiple result files
results = {}
for filename in ['eval_results_baseline.jsonl', 'eval_results_semantic.jsonl']:
    with open(filename) as f:
        results[filename] = [json.loads(line) for line in f]

# Create comparison DataFrame
comparison = []
for filename, data in results.items():
    valid = [r for r in data if 'error' not in r]
    comparison.append({
        'Method': filename.replace('eval_results_', '').replace('.jsonl', ''),
        'F1': sum(r['f1'] for r in valid) / len(valid) if valid else 0,
        'Precision': sum(r['precision'] for r in valid) / len(valid) if valid else 0,
        'Recall': sum(r['recall'] for r in valid) / len(valid) if valid else 0,
        'Success Rate': len(valid) / len(data),
    })

df = pd.DataFrame(comparison)
print(df.to_string(index=False))
```
