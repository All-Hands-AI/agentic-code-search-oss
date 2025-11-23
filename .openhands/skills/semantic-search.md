---
name: Semantic Code Search
---

# Semantic Code Search

Find code by meaning, not just keywords. Three MCP tools available:

## Tools

**`index_repository`** - Index the current workspace (call once)
```json
{
  "file_extensions": [".py"],
  "force_rebuild": false
}
```

**`semantic_search`** - Search by natural language
```json
{
  "query": "validation logic for email addresses",
  "n_results": 15,
  "return_content": true
}
```

**`get_index_stats`** - Check index status
```json
{}
```

## When to Use

**Semantic Search** - Finding code by what it does:
- ✅ "function that validates email format"
- ✅ "error handling for database connections"
- ✅ "code that calculates precision and recall"

**Bash/Grep** - Exact patterns:
- ✅ `grep -n 'def validate_email'`
- ✅ `rg 'class.*Validator' -t py`

## Strategy

1. **Semantic search** for candidates (n_results=15)
2. **Bash/grep** to verify specific patterns
3. **Return** identified files

## Tips

- Be descriptive: "validation logic for emails" not "function"
- Similarity scores: 0.9+ excellent, 0.7+ good, <0.5 check with grep
- Index persists - only rebuild if code changed significantly
- Increase n_results if first search misses files