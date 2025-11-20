"""
MCP Server for Semantic Code Search.

This server exposes semantic search capabilities through the Model Context Protocol (MCP),
allowing AI agents to perform vector-based code search.
"""

import json
import os
from pathlib import Path
from typing import Any

try:
    from mcp import Server, Tool
    from mcp.server import stdio_server
    from mcp.server.models import InitializationOptions
    from mcp.types import TextContent, EmbeddedResource
except ImportError:
    raise ImportError(
        "Please install MCP SDK: pip install mcp --break-system-packages"
    )

from src.tools.semantic_search import SemanticSearch


# Global state for the MCP server
server = Server("semantic-code-search")
indices = {}  # Store indices per repository


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="index_repository",
            description=(
                "Index a code repository for semantic search. "
                "This creates a vector index of all code files in the repository. "
                "Should be called once before searching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Absolute path to the repository to index",
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to index (e.g., ['.py', '.js'])",
                        "default": [".py"],
                    },
                    "force_rebuild": {
                        "type": "boolean",
                        "description": "Force rebuild the index even if it exists",
                        "default": False,
                    },
                },
                "required": ["repo_path"],
            },
        ),
        Tool(
            name="semantic_search",
            description=(
                "Search a repository using semantic similarity. "
                "Finds code based on natural language descriptions, not just keywords. "
                "More powerful than grep for finding code by meaning. "
                "Example: 'function that parses git diffs' or 'code for calculating precision and recall'"
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
                        "description": "Absolute path to the repository to search",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "return_content": {
                        "type": "boolean",
                        "description": "Whether to return full code content or just file paths",
                        "default": True,
                    },
                },
                "required": ["query", "repo_path"],
            },
        ),
        Tool(
            name="get_index_stats",
            description=(
                "Get statistics about an indexed repository, "
                "including number of indexed files and chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Absolute path to the repository",
                    },
                },
                "required": ["repo_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    try:
        if name == "index_repository":
            return await handle_index_repository(arguments)
        elif name == "semantic_search":
            return await handle_semantic_search(arguments)
        elif name == "get_index_stats":
            return await handle_get_index_stats(arguments)
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


async def handle_index_repository(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle repository indexing."""
    repo_path = Path(arguments["repo_path"]).resolve()
    file_extensions = arguments.get("file_extensions", [".py"])
    force_rebuild = arguments.get("force_rebuild", False)

    if not repo_path.exists():
        return [
            TextContent(
                type="text",
                text=f"Error: Repository path does not exist: {repo_path}",
            )
        ]

    # Create or get index
    repo_key = str(repo_path)
    if repo_key not in indices or force_rebuild:
        index = SemanticSearch(
            collection_name=f"code_index_{repo_path.name}",
            persist_directory=str(repo_path / ".vector_index"),
        )

        if force_rebuild:
            index.clear_index()

        # Index the repository
        stats = index.index_code_files(
            str(repo_path), file_extensions=file_extensions
        )
        indices[repo_key] = index

        result = (
            f"Successfully indexed repository: {repo_path}\n"
            f"Files indexed: {stats['indexed_files']}\n"
            f"Total chunks: {stats['total_chunks']}\n"
            f"Collection: {stats['collection_name']}"
        )
    else:
        index = indices[repo_key]
        stats = index.get_stats()
        result = (
            f"Repository already indexed: {repo_path}\n"
            f"Total documents: {stats['total_documents']}\n"
            f"Use force_rebuild=true to rebuild the index"
        )

    return [TextContent(type="text", text=result)]


async def handle_semantic_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle semantic search."""
    query = arguments["query"]
    repo_path = Path(arguments["repo_path"]).resolve()
    n_results = arguments.get("n_results", 10)
    return_content = arguments.get("return_content", True)

    if not repo_path.exists():
        return [
            TextContent(
                type="text",
                text=f"Error: Repository path does not exist: {repo_path}",
            )
        ]

    # Get or create index
    repo_key = str(repo_path)
    if repo_key not in indices:
        index = SemanticSearch(
            collection_name=f"code_index_{repo_path.name}",
            persist_directory=str(repo_path / ".vector_index"),
        )
        indices[repo_key] = index

        # Check if index exists
        stats = index.get_stats()
        if stats["total_documents"] == 0:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Repository not indexed. Please call index_repository first.",
                )
            ]
    else:
        index = indices[repo_key]

    # Perform search
    results = index.search(query, n_results=n_results)

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
        similarity = result["similarity_score"]
        file_path = result["file_path"]
        chunk_idx = result["chunk_index"]
        total_chunks = result["metadata"]["total_chunks"]

        output_lines.append(
            f"\n{i}. {file_path} (similarity: {similarity:.3f})"
        )
        output_lines.append(f"   Chunk {chunk_idx + 1}/{total_chunks}")

        if return_content:
            # Show content with limited preview
            content = result["content"]
            lines = content.split("\n")
            if len(lines) > 20:
                preview = "\n".join(lines[:20])
                output_lines.append(f"\n{preview}\n   ... ({len(lines)} total lines)")
            else:
                output_lines.append(f"\n{content}")

    # Add unique files summary
    unique_files = index.get_unique_files(results)
    output_lines.append(f"\n\nUnique files ({len(unique_files)}):")
    for file_path in unique_files:
        output_lines.append(f"  - {file_path}")

    result_text = "\n".join(output_lines)

    return [TextContent(type="text", text=result_text)]


async def handle_get_index_stats(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle getting index statistics."""
    repo_path = Path(arguments["repo_path"]).resolve()

    if not repo_path.exists():
        return [
            TextContent(
                type="text",
                text=f"Error: Repository path does not exist: {repo_path}",
            )
        ]

    # Get or create index
    repo_key = str(repo_path)
    if repo_key not in indices:
        index = SemanticSearch(
            collection_name=f"code_index_{repo_path.name}",
            persist_directory=str(repo_path / ".vector_index"),
        )
        indices[repo_key] = index
    else:
        index = indices[repo_key]

    stats = index.get_stats()

    result = (
        f"Index Statistics for {repo_path}:\n"
        f"Collection: {stats['collection_name']}\n"
        f"Total documents: {stats['total_documents']}\n"
        f"Embedding model: {stats['embedding_model']}\n"
        f"Persist directory: {stats['persist_directory']}"
    )

    return [TextContent(type="text", text=result)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="semantic-code-search",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
