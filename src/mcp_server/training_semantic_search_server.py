"""
Lightweight MCP server for training that delegates to EmbeddingService.

This MCP server does NOT load embedding models locally.
Instead, it calls a shared Ray actor that has models pre-loaded.

Use this during training for massive efficiency gains.
"""

import json
import os
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Optional
import ray

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, EmbeddedResource
except ImportError as e:
    raise ImportError(
        f"Please install MCP SDK: uv pip install mcp fastmcp\nError: {e}"
    )


# Global state
server = Server("semantic-code-search-training")
embedding_service = None


def get_workspace_path() -> str:
    """Get workspace path from environment variable."""
    workspace = os.getenv("WORKSPACE_PATH")
    if not workspace:
        raise ValueError(
            "WORKSPACE_PATH environment variable not set. "
            "Please configure the MCP server with the workspace path."
        )
    return workspace


def get_repo_info(repo_path: Path) -> tuple[str, str]:
    """Extract repo name and commit hash from git repository."""
    try:
        # Get current commit
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()

        # Get remote URL to extract repo name
        result = subprocess.run(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        )
        url = result.stdout.strip()

        # Parse repo name from URL
        if "github.com" in url:
            parts = url.rstrip(".git").split("/")
            repo_name = "/".join(parts[-2:])
        else:
            repo_name = repo_path.name

        return repo_name, commit
    except Exception:
        # Fallback: use directory name
        return repo_path.name, "unknown"


def get_embedding_service_ref():
    """Get reference to shared embedding service."""
    global embedding_service

    if embedding_service is None:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray_address = os.getenv("RAY_ADDRESS", "auto")
            try:
                ray.init(address=ray_address, ignore_reinit_error=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to Ray cluster at {ray_address}: {e}"
                )

        try:
            # Connect to existing service
            embedding_service = ray.get_actor("embedding_service")
        except ValueError:
            raise ValueError(
                "Embedding service not found. "
                "Initialize it at training start with get_embedding_service()"
            )

    return embedding_service


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="semantic_search",
            description=(
                "Search a repository using semantic similarity (training-optimized). "
                "Uses shared embedding service for fast inference. "
                "Requires pre-computed indices (run preindex_swebench.py first)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what you're looking for",
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository (optional, defaults to current workspace)",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "semantic_search":
            return await handle_semantic_search(arguments)
        else:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Unknown tool '{name}'",
                )
            ]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}",
            )
        ]


async def handle_semantic_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle semantic search using shared embedding service."""
    query = arguments["query"]
    repo_path = Path(arguments.get("repo_path") or get_workspace_path()).resolve()
    n_results = arguments.get("n_results", 10)

    if not repo_path.exists():
        return [
            TextContent(
                type="text",
                text=f"Error: Repository path does not exist: {repo_path}",
            )
        ]

    # Get repo info
    repo_name, commit = get_repo_info(repo_path)

    # Get embedding service reference
    service = get_embedding_service_ref()

    # Delegate to service (FAST - no model loading!)
    try:
        results = ray.get(service.search.remote(query, repo_name, commit, n_results))
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Error: {str(e)}\nMake sure to run pre-indexing: python preindex_swebench.py",
            )
        ]

    if not results:
        return [
            TextContent(
                type="text",
                text=f"No results found for query: {query}",
            )
        ]

    # Format results
    output_lines = [f"Found {len(results)} relevant code chunks for: '{query}'\n"]

    for i, result in enumerate(results, 1):
        similarity = result.get("rerank_score", result.get("similarity_score", 0))
        file_path = result["file_path"]
        chunk_idx = result["chunk_index"]
        total_chunks = result["metadata"]["total_chunks"]

        output_lines.append(
            f"\n{i}. {file_path} (similarity: {similarity:.3f})"
        )
        output_lines.append(f"   Chunk {chunk_idx + 1}/{total_chunks}")

        # Show content preview
        content = result["content"]
        lines = content.split("\n")
        if len(lines) > 20:
            preview = "\n".join(lines[:20])
            output_lines.append(f"\n{preview}\n   ... ({len(lines)} total lines)")
        else:
            output_lines.append(f"\n{content}")

    # Add unique files summary
    unique_files = list(set(r["file_path"] for r in results))
    output_lines.append(f"\n\nUnique files ({len(unique_files)}):")
    for file_path in unique_files:
        output_lines.append(f"  - {file_path}")

    result_text = "\n".join(output_lines)

    return [TextContent(type="text", text=result_text)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())