# Running Semantic Search with Local vLLM

This guide shows how to run the semantic code search agent with a locally served LLM using vLLM.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Agent (openhands-sdk)                                  │
│    ├─ LLM Client ──────────────> vLLM Server (GPU)     │
│    └─ MCP Client ──────────────> MCP Server (subprocess)│
│         └─ Semantic Search ───> ChromaDB (local)        │
└─────────────────────────────────────────────────────────┘
```

**All components run locally - no external API calls needed!**

## Setup

### 1. Start vLLM Server

```bash
# Install vLLM (already in dependencies)
uv sync

# Start vLLM with your chosen model
vllm serve deepseek-ai/deepseek-coder-33b-instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --dtype auto
```

**Common Models for Code:**
- `deepseek-ai/deepseek-coder-33b-instruct` - Great for code tasks
- `codellama/CodeLlama-34b-Instruct-hf` - Meta's code LLM
- `WizardLM/WizardCoder-33B-V1.1` - Strong code generation
- `Qwen/Qwen2.5-Coder-32B-Instruct` - Latest Qwen coder

**vLLM Options:**
- `--tensor-parallel-size`: Number of GPUs (1 for single GPU)
- `--max-model-len`: Context length (lower if OOM)
- `--dtype auto`: Automatic dtype selection (bfloat16, float16, etc.)
- `--gpu-memory-utilization 0.9`: Increase if you have enough VRAM

### 2. Verify vLLM is Running

```bash
# Test the endpoint
curl http://localhost:8000/v1/models

# Should return something like:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "deepseek-ai/deepseek-coder-33b-instruct",
#       "object": "model",
#       ...
#     }
#   ]
# }
```

### 3. Run the Agent

```bash
# Run the vLLM example
uv run python examples/mcp_agent_with_vllm.py
```

## Configuration Options

### Low VRAM Setup (< 24GB)

For smaller GPUs, use quantized models or smaller models:

```bash
# AWQ quantized model (4-bit)
vllm serve TheBloke/deepseek-coder-33B-instruct-AWQ \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
```

### Multi-GPU Setup

```bash
# Tensor parallelism across 2 GPUs
vllm serve deepseek-ai/deepseek-coder-33b-instruct \
    --tensor-parallel-size 2 \
    --max-model-len 16384
```

### Custom Model Settings in Agent

```python
llm = LLM(
    model="deepseek-coder-33b-instruct",
    base_url="http://localhost:8000/v1",
    api_key=SecretStr("EMPTY"),

    # Sampling parameters
    temperature=0.1,        # Lower = more deterministic
    top_p=0.95,            # Nucleus sampling
    max_output_tokens=4096,

    # Request handling
    timeout=180,           # 3 minutes for complex tasks
    num_retries=3,
)
```

## Comparing Performance: Local vs API

| Aspect | vLLM (Local) | Claude API |
|--------|--------------|------------|
| **Cost** | $0 (hardware only) | ~$3-15 per 1M tokens |
| **Latency** | 10-50ms (local network) | 200-1000ms (internet) |
| **Privacy** | 100% private | Data sent to Anthropic |
| **Setup** | Requires GPU (8GB+ VRAM) | Just API key |
| **Model Quality** | Depends on model size | State-of-the-art |
| **Throughput** | Limited by GPU | Limited by rate limits |

## Troubleshooting

### vLLM Server Won't Start

**Out of Memory:**
```bash
# Reduce context length
vllm serve your-model --max-model-len 4096

# Or use a smaller model
vllm serve deepseek-ai/deepseek-coder-6.7b-instruct
```

**CUDA Errors:**
```bash
# Check GPU availability
nvidia-smi

# Reinstall vLLM
uv pip install --force-reinstall vllm==0.11.0
```

### Agent Can't Connect to vLLM

```python
# Test connection manually
import httpx
response = httpx.get("http://localhost:8000/v1/models")
print(response.json())

# If this fails, check vLLM is running and port is correct
```

### MCP Server Issues

```bash
# Test MCP server directly
uv run python src/mcp_server/semantic_search_server.py

# Check if chromadb dependencies are installed
uv pip list | grep chromadb
```

## Performance Tips

1. **Pre-index repositories** before agent runs:
   ```bash
   uv run python scripts/preindex_repos.py /path/to/repo
   ```

2. **Adjust chunk size** for better retrieval:
   ```python
   index = SemanticSearch(max_chunk_size=256)  # Smaller chunks
   ```

3. **Use better embeddings** (optional):
   ```python
   index = SemanticSearch(
       embedding_model_name="BAAI/bge-large-en-v1.5",  # Better than jina
       reranker_model_name="BAAI/bge-reranker-v2-m3"
   )
   ```

4. **Enable prompt caching** in vLLM:
   ```bash
   vllm serve your-model --enable-prefix-caching
   ```

## Resource Requirements

**Minimal Setup:**
- GPU: 1x RTX 3090 (24GB VRAM)
- Model: DeepSeek-Coder 6.7B
- Context: 8K tokens
- Expected speed: ~50 tokens/sec

**Recommended Setup:**
- GPU: 2x RTX 4090 (48GB total VRAM)
- Model: DeepSeek-Coder 33B
- Context: 16K tokens
- Expected speed: ~100 tokens/sec

**High-Performance Setup:**
- GPU: 4x H100 (320GB total VRAM)
- Model: DeepSeek-Coder 33B or larger
- Context: 32K tokens
- Expected speed: ~200+ tokens/sec
