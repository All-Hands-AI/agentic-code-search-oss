---
name: Semantic Code Search
trigger: always
mcp_servers:
  - semantic-code-search
---

# Semantic Code Search Skill

You have access to **semantic code search** through MCP tools. This allows you to find code by meaning, not just keywords.

## Available MCP Tools

### 1. `index_repository`
Index a repository for semantic search (call once per repository).

**Parameters:**
- `repo_path` (string, required): Absolute path to repository
- `file_extensions` (array, optional): File types to index (default: [".py"])
- `force_rebuild` (boolean, optional): Rebuild existing index (default: false)

**When to use:** On first access to a new repository, before searching.

**Example:**
```
index_repository({
  "repo_path": "/workspace/django_django-12345",
  "file_extensions": [".py"],
  "force_rebuild": false
})
```

### 2. `semantic_search`
Search code using natural language descriptions.

**Parameters:**
- `query` (string, required): Natural language description of what you're looking for
- `repo_path` (string, required): Absolute path to repository
- `n_results` (integer, optional): Number of results to return (default: 10, max: 50)
- `return_content` (boolean, optional): Include code content in results (default: true)

**When to use:** Finding code by conceptual description rather than exact keywords.

**Example:**
```
semantic_search({
  "query": "function that validates email addresses and checks format",
  "repo_path": "/workspace/django_django-12345",
  "n_results": 15,
  "return_content": true
})
```

### 3. `get_index_stats`
Get statistics about an indexed repository.

**Parameters:**
- `repo_path` (string, required): Absolute path to repository

**When to use:** Checking if a repository is indexed, or debugging.

**Example:**
```
get_index_stats({
  "repo_path": "/workspace/django_django-12345"
})
```

## Recommended Search Strategy

For code localization tasks, follow this workflow:

### Step 1: Semantic Search (Broad Exploration)
Start with semantic search to find conceptually relevant files:
```
semantic_search({
  "query": "validation logic for user input and email format checking",
  "repo_path": "/workspace/repo",
  "n_results": 15
})
```

This returns ~10-15 candidate files with similarity scores.

### Step 2: Bash/Grep (Verification)
Use bash/grep to verify specific patterns in candidate files:
```bash
bash("grep -n 'def validate_email' path/to/candidate.py")
bash("rg 'class.*Validator' -t py path/to/")
```

### Step 3: Return Results
Once you've identified relevant files, use the `result` tool:
```
result(["path/to/file1.py", "path/to/file2.py"])
```

## When to Use Each Tool

### Use Semantic Search When:
- Problem description is conceptual: "validation logic", "error handling", "data processing"
- Looking for code that does something, not exact function names
- Exploring unfamiliar codebase
- Keywords might vary (e.g., "validate" vs "check" vs "verify")

**Example queries:**
- ✅ "function that calculates precision and recall metrics"
- ✅ "code that parses git diffs and extracts file paths"
- ✅ "validation logic for email addresses"
- ✅ "error handling for database connections"

### Use Bash/Grep When:
- Need exact pattern matching: specific function names, class names
- Verifying semantic search results
- Looking for imports, specific strings, regex patterns
- Exploring file structure (ls, find)

**Example commands:**
- ✅ `bash("rg 'def calculate_f1' -t py")`
- ✅ `bash("grep -r 'import django' --include='*.py'")`
- ✅ `bash("find . -name '*validator*.py'")`

## Best Practices

### 1. Be Descriptive in Semantic Queries
❌ Bad: "function"
✅ Good: "function that validates email format and checks domain"

❌ Bad: "code"
✅ Good: "code that handles authentication errors and retries"

### 2. Combine Tools for Best Results
```
1. semantic_search("validation logic") → Get candidates
2. bash("grep -n 'def validate' file1.py file2.py") → Verify
3. result(["file1.py", "file2.py"]) → Return answer
```

### 3. Adjust n_results Based on Task
- Simple tasks: 5-10 results
- Complex tasks: 15-20 results
- If first search misses files: increase to 30-50

### 4. Index Once Per Repository
The index persists across runs. Only call `index_repository` if:
- First time accessing this repository
- Repository code has changed significantly
- Want to force rebuild

## Understanding Results

Semantic search returns results with **similarity scores** (0.0 to 1.0):
- **0.9+**: Highly relevant, almost certainly correct
- **0.7-0.9**: Very relevant, likely correct
- **0.5-0.7**: Somewhat relevant, needs verification
- **< 0.5**: Possibly not relevant

Use similarity scores to prioritize which files to examine first.

## Performance Notes

- **First search**: May take 1-2 seconds (loading index)
- **Subsequent searches**: <1 second (index cached)
- **Indexing**: 10-60 seconds depending on repository size
- **Memory usage**: ~200-400 MB per indexed repository

## Troubleshooting

### "Repository not indexed"
Call `index_repository` before searching.

### "No results found"
Try:
1. Broader query: "validation" instead of "email validation for user accounts"
2. More results: increase n_results to 20-30
3. Different phrasing: "error handling" instead of "exception management"

### "Low similarity scores"
Repository might not contain the code you're looking for. Use bash/grep as fallback.

## Examples

### Example 1: Finding Validation Code
```
# Step 1: Semantic search
semantic_search({
  "query": "email validation function that checks format and domain",
  "repo_path": "/workspace/repo",
  "n_results": 10
})

# Step 2: Verify results
bash("grep -n 'def.*validate.*email' path/to/validators.py")

# Step 3: Return
result(["path/to/validators.py", "path/to/models.py"])
```

### Example 2: Finding Test Code
```
semantic_search({
  "query": "unit tests for user authentication and login",
  "repo_path": "/workspace/repo",
  "n_results": 15
})
```

### Example 3: Finding Database Code
```
semantic_search({
  "query": "database connection pooling and query execution",
  "repo_path": "/workspace/repo",
  "n_results": 10
})
```

## Summary

Semantic search is your **primary exploration tool** for code localization. Use it to:
- Find code by what it does, not what it's called
- Discover relevant files quickly
- Handle varied terminology and concepts

Combine with bash/grep for verification, and you'll have a powerful code localization system! 🚀
