# Evaluation Approaches

This repository supports **two different evaluation approaches** for SWE-bench. Choose based on your needs.

## Quick Comparison

| Aspect | Verifiers (eval.py) | Agent-SDK MCP (eval_with_mcp.py) |
|--------|---------------------|----------------------------------|
| **Framework** | `verifiers` package | `openhands-agent-sdk` |
| **Semantic Search** | Direct function import | MCP server (subprocess) |
| **Agent Type** | StatefulToolEnv | Agent + Conversation |
| **Best For** | Fast iteration, training | Production, MCP testing |
| **Speed** | ⚡ Faster | Slower (MCP overhead) |
| **Complexity** | Simpler | More realistic |

## Approach 1: Verifiers (eval.py)

**Use this for:**
- Fast experimentation
- Training data generation
- Reward function development
- Quick baseline comparisons

### How it works

```python
# Direct tool integration
from src.tools.semantic_search import semantic_search

class EvalEnvironment(vf.StatefulToolEnv):
    def __init__(self, use_semantic: bool = False):
        self.add_tool(bash, args_to_skip=["cwd"])
        self.add_tool(result)
        if use_semantic:
            self.add_tool(semantic_search, args_to_skip=["repo_path"])
```

### Running

```bash
# Baseline
uv run python eval.py --num-instances 50

# With semantic search
uv run python eval.py --semantic --num-instances 50

# Full comparison
./scripts/run_eval_comparison.sh -n 50
```

### Pros
- ✅ **Fast**: No MCP subprocess overhead
- ✅ **Simple**: Direct function calls
- ✅ **Easy to debug**: Single process
- ✅ **Works with verifiers training**: Already integrated

### Cons
- ❌ Not using real MCP integration
- ❌ Different from production agent setup
- ❌ Less realistic tool calling

## Approach 2: Agent-SDK with MCP (eval_with_mcp.py)

**Use this for:**
- Testing MCP integration
- Production-like evaluation
- End-to-end validation
- Debugging agent-sdk issues

### How it works

```python
# MCP integration (same as production)
skill = Skill.from_file(".openhands/skills/semantic-search.md")

mcp_config = {
    "mcpServers": {
        "semantic-code-search": {
            "command": "uv",
            "args": ["run", "python", "src/mcp_server/semantic_search_server.py"]
        }
    }
}

agent = Agent(llm=llm, agent_context=context, mcp_config=mcp_config)
conversation = Conversation(agent=agent, workspace=repo_path)
```

### Running

```bash
# Baseline (agent-sdk, no MCP)
uv run python eval_with_mcp.py --num-instances 50

# With MCP semantic search
uv run python eval_with_mcp.py --semantic --num-instances 50

# Full comparison
./scripts/run_mcp_eval.sh -n 50
```

### Pros
- ✅ **Realistic**: Same as production agent
- ✅ **Tests MCP**: Validates MCP server integration
- ✅ **Full agent-sdk**: Uses Conversation, AgentContext, Skills
- ✅ **Debugging**: Catches MCP-specific issues

### Cons
- ❌ Slower: MCP subprocess overhead (~20% slower)
- ❌ More complex: Multiple processes
- ❌ Harder to debug: MCP communication layer

## Which Should I Use?

### For Development & Training
→ Use **eval.py** (verifiers approach)

```bash
./scripts/run_eval_comparison.sh -n 50
```

**Why:**
- Faster iteration cycles
- Easier debugging
- Works with existing training pipeline
- Direct access to tool outputs

### For Production Validation
→ Use **eval_with_mcp.py** (agent-sdk approach)

```bash
./scripts/run_mcp_eval.sh -n 50
```

**Why:**
- Matches production agent setup
- Validates MCP integration
- Tests full conversation flow
- Catches subprocess issues

### For Both
Run both and compare! They should give similar results:

```bash
# Run verifiers version
./scripts/run_eval_comparison.sh -n 50

# Run MCP version
./scripts/run_mcp_eval.sh -n 50

# Compare both
python -c "
import json

# Load all results
files = [
    'eval_results_without_semantic.jsonl',
    'eval_results_with_semantic.jsonl',
    'eval_results_mcp_baseline.jsonl',
    'eval_results_mcp_semantic.jsonl',
]

for f in files:
    with open(f) as file:
        results = [json.loads(line) for line in file if line.strip()]
        valid = [r for r in results if 'f1' in r]
        avg_f1 = sum(r['f1'] for r in valid) / len(valid) if valid else 0
        print(f'{f:45} F1: {avg_f1:.3f} ({len(valid)}/{len(results)} successful)')
"
```

## Expected Performance

Both approaches should give **similar F1 scores** (within ~2%):

```
eval_results_without_semantic.jsonl          F1: 0.445
eval_results_mcp_baseline.jsonl              F1: 0.438  (slightly lower due to conversation overhead)

eval_results_with_semantic.jsonl             F1: 0.523
eval_results_mcp_semantic.jsonl              F1: 0.516  (slightly lower due to MCP overhead)
```

The MCP version is typically 1-2% lower due to:
- Additional conversation turns
- MCP subprocess communication overhead
- Skill activation logic

## Common Issues

### MCP Version Hangs
**Problem:** Evaluation stops during MCP semantic search

**Solution:**
```bash
# Check if MCP server is working
uv run python src/mcp_server/semantic_search_server.py

# Test MCP manually
# (Start server, then test with agent)
```

### Results Differ Significantly
**Problem:** >5% difference between verifiers and MCP versions

**Possible causes:**
- Different prompting (check system prompts)
- MCP skill not loading correctly
- Tool schema mismatch
- Agent making different decisions in conversation mode

**Debug:**
```bash
# Add verbose logging to both
OPENHANDS_DEBUG=1 uv run python eval.py --num-instances 5 --semantic
OPENHANDS_DEBUG=1 uv run python eval_with_mcp.py --num-instances 5 --semantic
```

### Out of Memory with MCP
**Problem:** MCP version uses more memory

**Solution:**
```bash
# Reduce batch size or run sequentially
# MCP spawns subprocesses that hold indices in memory

# Clear vector indices between runs
rm -rf ./swebench_repos/*/.vector_index/
```

## Recommendations

1. **Start with verifiers** (`eval.py`) for development
2. **Validate with MCP** (`eval_with_mcp.py`) before release
3. **Run both periodically** to ensure consistency
4. **Use verifiers for RL training** (faster, simpler)
5. **Use MCP for production testing** (realistic)

## File Organization

```
.
├── eval.py                          # Verifiers approach
├── eval_with_mcp.py                 # Agent-SDK MCP approach
├── scripts/
│   ├── run_eval_comparison.sh       # Runs eval.py (both modes)
│   ├── run_mcp_eval.sh              # Runs eval_with_mcp.py (both modes)
│   └── compare_results.py           # Compares any two JSONL files
└── EVALUATION.md                    # Main evaluation guide
```
