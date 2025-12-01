# src/tools/semantic_search.py
import os
from pathlib import Path
from typing import Optional, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Fix tree_sitter import for newer versions
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
except ImportError:
    # Fallback: disable tree-sitter chunking
    print("Warning: tree_sitter_python not installed. Using simple line-based chunking.")
    PY_LANGUAGE = None
    parser = None

class SemanticSearch:
    """Vector index + reranker for semantic code search (fully local)."""

    def __init__(
        self,
        embedding_model_name: str = "jinaai/jina-code-embeddings-0.5b",
        reranker_model_name: Optional[str] = "jinaai/jina-reranker-v3",  # CHANGED: Disable by default for speed
        collection_name: str = "code_index",
        persist_directory: str = "./.vector_index",
        device: str = "cpu",  # CHANGED: Default to CPU
        max_chunk_size: int = 512,
        num_threads: int = 8,  # NEW: CPU threading
    ):
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.max_chunk_size = max_chunk_size
        self.device = device
        self.num_threads = num_threads

        # NEW: Enable multi-threading for CPU
        if device == "cpu":
            import torch
            torch.set_num_threads(num_threads)

        # Initialize models
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        
        # NEW: Optimize for speed
        if device == "cpu":
            self.embedder.to("cpu")
        
        if reranker_model_name:
            self.reranker = CrossEncoder(reranker_model_name, device=device)
            if device == "cpu":
                # Optimize reranker for CPU
                import torch
                self.reranker.model.to("cpu")
        else:
            self.reranker = None

        # Initialize ChromaDB PersistentClient
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_code_files(
        self,
        repo_path: str,
        file_extensions: Optional[List[str]] = None,
        batch_size: int = 32,
        exclude_patterns: Optional[List[str]] = None,
    ) -> dict:
        if file_extensions is None:
            file_extensions = [".py"]
        
        if exclude_patterns is None:
            # FIXED: More specific patterns to avoid false positives
            exclude_patterns = [
                "__pycache__", ".pytest_cache", 
                "node_modules", ".venv", "venv", "env", ".git",
                ".tox", ".eggs", "dist", "build",
            ]

        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Collect files
        files_to_index = []
        for ext in file_extensions:
            files_to_index.extend(repo_path.rglob(f"*{ext}"))

        print(f"[SemanticSearch] Found {len(files_to_index)} total files")

        # IMPROVED: More careful filtering
        def should_exclude(file_path: Path) -> bool:
            """Check if file should be excluded based on patterns."""
            # Get path relative to repo root
            try:
                relative_path = file_path.relative_to(repo_path)
            except ValueError:
                return True
            
            path_parts = relative_path.parts
            path_str = str(relative_path).lower()
            file_name = file_path.name.lower()
            
            for pattern in exclude_patterns:
                pattern_lower = pattern.lower()
                
                # Check if pattern matches any directory component exactly
                if pattern_lower in path_parts:
                    return True
            
            # ADDITIONAL: Exclude test directories (but not files containing "test" in parent paths)
            test_dir_patterns = ["test", "tests", "testing"]
            for part in path_parts[:-1]:  # Check all directories, not the filename
                if part.lower() in test_dir_patterns:
                    return True
            
            # ADDITIONAL: Exclude test files (files starting with test_ or ending with _test.py)
            if file_name.startswith("test_") or file_name.endswith("_test.py"):
                return True
            
            # ADDITIONAL: Exclude docs and examples directories
            if "docs" in path_parts or "examples" in path_parts:
                return True
            
            return False
        
        original_count = len(files_to_index)
        files_to_index = [f for f in files_to_index if not should_exclude(f)]
        excluded_count = original_count - len(files_to_index)

        print(f"[SemanticSearch] After filtering: {len(files_to_index)} files to index ({excluded_count} excluded)")
        
    def search(self, query: str, n_results: int = 10, filter_metadata: Optional[dict] = None, use_reranker: bool = True) -> list[dict]:
        """Search with optional reranking."""
        # Generate query embedding
        query_emb = self.embedder.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Retrieve more candidates for reranking
        n_retrieve = n_results * 5 if (self.reranker) else n_results
        
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_retrieve,
            where=filter_metadata,
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "file_path": results["metadatas"][0][i]["file_path"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "content": results["documents"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i]
                })

        # Rerank if enabled and reranker exists
        if self.reranker and formatted_results:
            texts = [r["content"] for r in formatted_results]
            pairs = [(query, text) for text in texts]
            
            try:
                # FIX: Use batch_size=1 to avoid padding token issues
                rerank_scores = self.reranker.predict(
                    pairs, 
                    batch_size=1,  # CHANGED: Force batch_size=1 for stability
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                for i, score in enumerate(rerank_scores):
                    formatted_results[i]["rerank_score"] = float(score)

                formatted_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            except Exception as e:
                print(f"Warning: Reranking failed: {e}, using similarity scores only")
                # Fall back to similarity scores if reranking fails
        
        return formatted_results[:n_results]

    # ---------------------------
    # Utilities
    # ---------------------------
    def get_unique_files(self, search_results: list[dict]) -> list[str]:
        seen = set()
        unique_files = []
        for r in search_results:
            fp = r["file_path"]
            if fp not in seen:
                seen.add(fp)
                unique_files.append(fp)
        return unique_files

    def _chunk_text(self, text: str, max_size: int) -> list[str]:
        """Chunk text using tree-sitter if available, else line-based."""
        
        # Fallback to simple chunking if tree-sitter not available
        if parser is None:
            return self._chunk_lines_simple(text, max_size)
        
        try:
            tree = parser.parse(bytes(text, "utf8"))
            root_node = tree.root_node

            chunks = []

            # Helper: chunk text by lines when too large
            def chunk_lines(block_text, max_size):
                lines = block_text.split("\n")
                out_chunks = []
                current_chunk = []
                current_size = 0
                for line in lines:
                    line_size = len(line) + 1  # +1 for newline
                    if current_size + line_size > max_size and current_chunk:
                        out_chunks.append("\n".join(current_chunk))
                        current_chunk = [line]
                        current_size = line_size
                    else:
                        current_chunk.append(line)
                        current_size += line_size
                if current_chunk:
                    out_chunks.append("\n".join(current_chunk))
                return out_chunks

            # Extract function and class definitions
            for node in root_node.children:
                if node.type in ("function_definition", "class_definition"):
                    start, end = node.start_byte, node.end_byte
                    block = text[start:end]
                    if len(block) <= max_size:
                        chunks.append(block)
                    else:
                        chunks.extend(chunk_lines(block, max_size))

            # Handle code between functions/classes
            covered_ranges = [
                (node.start_byte, node.end_byte)
                for node in root_node.children
                if node.type in ("function_definition", "class_definition")
            ]
            
            last = 0
            for start, end in sorted(covered_ranges):
                if last < start:
                    leftover = text[last:start]
                    if leftover.strip():
                        chunks.extend(chunk_lines(leftover, max_size))
                last = end
            
            if last < len(text):
                leftover = text[last:]
                if leftover.strip():
                    chunks.extend(chunk_lines(leftover, max_size))

            return chunks
            
        except Exception as e:
            # Fallback if tree-sitter parsing fails
            print(f"Warning: tree-sitter parsing failed, using simple chunking: {e}")
            return self._chunk_lines_simple(text, max_size)
    
    def _chunk_lines_simple(self, text: str, max_size: int) -> list[str]:
        """Simple line-based chunking fallback."""
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            if current_size + line_size > max_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks

    def clear_index(self):
        """Clear the entire index."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory
        }


# Tool function for verifiers integration
def semantic_search(
    query: str,
    repo_path: str,
    n_results: int = 10,
    rebuild_index: bool = False,
) -> str:
    """
    Search code semantically using natural language.
    
    Args:
        query: Natural language description
        repo_path: Path to repository
        n_results: Number of results
        rebuild_index: Force rebuild
    
    Returns:
        Formatted string with results
    """
    from pathlib import Path
    
    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists():
        return f"Error: Repository not found: {repo_path}"
    
    # Create index
    index = SemanticSearch(
        collection_name=f"code_index_{repo_path_obj.name}",
        persist_directory=str(repo_path_obj / ".vector_index"),
    )
    
    # Check if needs indexing
    stats = index.get_stats()
    if stats["total_documents"] == 0 or rebuild_index:
        print(f"Indexing {repo_path}...")
        index.index_code_files(str(repo_path))
    
    # Search
    results = index.search(query, n_results=n_results)
    
    if not results:
        return f"No results found for query: {query}"
    
    # Format output
    unique_files = index.get_unique_files(results)
    output = f"Found {len(unique_files)} relevant files:\n\n"
    for file in unique_files:
        output += f"- {file}\n"
    
    return output