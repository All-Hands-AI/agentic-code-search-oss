# Vector Search Integration

This adds semantic code search capabilities to the agent through MCP (Model Context Protocol).

## What's Added

### Core Components

1. **Semantic Search Tool** (`src/tools/semantic_search.py`)
   - Vector-based code search using ChromaDB and sentence-transformers
   - Embeds code using `jinaai/jina-code-embeddings-0.5b`
   - Optional reranking with `jinaai/jina-reranker-v3`
   - Tree-sitter based code chunking for better results

2. **MCP Server** (`src/mcp_server/semantic_search_server.py`)
   - Exposes semantic search via Model Context Protocol
   - Provides three tools: `index_repository`, `semantic_search`, `get_index_stats`
   - Runs as subprocess spawned by agent-sdk

3. **Skill Definition** (`.openhands/skills/semantic-search.md`)
   - Configures MCP integration for agents
   - Contains tool documentation and search strategy guidance

### Dependencies

Added to `pyproject.toml`:
- `chromadb>=0.5.0` - Local vector database
- `sentence-transformers>=3.0.0` - Embedding models
- `tree-sitter>=0.25.2` - Code parsing
- `tree-sitter-python>=0.23.0` - Python language support

## Usage

### With Agent-SDK

```python
import asyncio
from openhands.sdk import LLM, Agent, Conversation, AgentContext, Skill
from pydantic import SecretStr

async def main():
    # Load semantic search skill
    skill = Skill.from_file(".openhands/skills/semantic-search.md")

    # Configure agent with skill
    llm = LLM(model="claude-sonnet-4-5", api_key=SecretStr("your-key"))
    context = AgentContext(skills=[skill])
    agent = Agent(llm=llm, agent_context=context)

    # Run conversation
    conversation = Conversation(agent=agent, workspace="/path/to/repo")
    conversation.send_message("Find code related to authentication")
    await conversation.run()

    print(conversation.agent_final_response())

asyncio.run(main())
```

### With Local vLLM

```python
# Configure LLM to use local vLLM server
llm = LLM(
    model="deepseek-ai/deepseek-coder-33b-instruct",
    base_url="http://localhost:8000/v1",
    api_key=SecretStr("EMPTY"),
)

# Rest is the same
```

See `examples/mcp_agent_example.py` for complete example.

### Direct Tool Usage

```python
from src.tools.semantic_search import SemanticSearch

# Create index
index = SemanticSearch(
    collection_name="my_repo",
    persist_directory="./.vector_index",
)

# Index repository
index.index_code_files("/path/to/repo", file_extensions=[".py"])

# Search
results = index.search("function that validates email addresses", n_results=10)
for result in results:
    print(f"{result['file_path']}: {result['similarity_score']:.3f}")
```

## How It Works

### Indexing
1. Scans repository for code files
2. Chunks files using tree-sitter (code-aware) or line-based fallback
3. Generates embeddings using sentence-transformers
4. Stores in ChromaDB (local, persistent)

### Searching
1. Embeds query using same model
2. Retrieves top candidates via vector similarity
3. Optionally reranks using cross-encoder
4. Returns ranked results with similarity scores

### MCP Integration
- Agent-SDK automatically spawns MCP server when skill is loaded
- Server communicates via stdio (standard MCP protocol)
- Tools are exposed through normal agent tool calling
- Vector indices persist across runs (`.vector_index/` directory)

## Storage

Vector indices are stored locally:
```
/path/to/repo/.vector_index/          # Repository-specific index
~/.cache/huggingface/hub/              # Downloaded models (~500MB)
```

## Performance

**First Time:**
- Downloads embedding models (~500MB, one-time)
- Indexes repository (depends on size, ~1-2min for large repos)

**Subsequent Uses:**
- Loads existing index (fast)
- Search queries: ~50-200ms

**Resource Usage:**
- RAM: ~2-3GB for models + index
- Disk: ~50-100MB per indexed repository
- No GPU required (CPU-only)

## Examples

### Basic Example
```bash
# Run with Claude API
export LLM_API_KEY="your-anthropic-api-key"
uv run python examples/mcp_agent_example.py
```

### Local vLLM Example
```bash
# Start vLLM server
vllm serve deepseek-ai/deepseek-coder-33b-instruct \
    --host 0.0.0.0 \
    --port 8000

# Run agent
uv run python examples/mcp_agent_with_vllm.py
```

See `examples/vllm_setup.md` for detailed vLLM setup guide.

## Architecture

```
Agent (openhands-sdk)
  │
  ├─> Skill (.openhands/skills/semantic-search.md)
  │   └─> MCP Config (embedded)
  │
  ├─> MCP Client (agent-sdk)
  │   └─> Spawns subprocess
  │
  └─> MCP Server (src/mcp_server/semantic_search_server.py)
      └─> SemanticSearch (src/tools/semantic_search.py)
          ├─> ChromaDB (local vector DB)
          └─> SentenceTransformers (embeddings)
```

## Testing

```bash
# Run tests
uv run pytest tests/test_semantic_search.py -v
```

## Troubleshooting

### Models Not Downloading
```bash
# Manually download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-code-embeddings-0.5b')"
```

### Index Issues
```bash
# Clear and rebuild index
rm -rf /path/to/repo/.vector_index/
# Then re-index using the tool
```

### MCP Server Not Starting
```bash
# Test MCP server directly
uv run python src/mcp_server/semantic_search_server.py
# Should show JSON-RPC messages on stdin/stdout
```

## Integration with Existing Code

This integrates cleanly with the existing codebase:
- No changes to existing tools (`bash`, `result`)
- No changes to existing prompts or rewards
- Optional: Can be used alongside traditional grep/search
- Backward compatible: Works with or without semantic search enabled
