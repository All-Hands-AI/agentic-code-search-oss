import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


class CodebaseTools:
    """Experimental tools for code navigation and search"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.valid_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.ts', '.jsx', '.tsx'}
        self.scroll_states = {}
        
        # Definition patterns by language
        self.def_patterns = {
            '.py': [r'^class\s+{kw}\b', r'^def\s+{kw}\s*\(', r'^{kw}\s*='],
            '.js': [r'class\s+{kw}\b', r'function\s+{kw}\s*\(', r'const\s+{kw}\s*=', r'let\s+{kw}\s*='],
            '.ts': [r'class\s+{kw}\b', r'function\s+{kw}\s*\(', r'const\s+{kw}\s*[:=]', r'interface\s+{kw}\b'],
            '.java': [r'class\s+{kw}\b', r'interface\s+{kw}\b', r'\w+\s+{kw}\s*\('],
            '.go': [r'func\s+{kw}\s*\(', r'type\s+{kw}\s+struct', r'type\s+{kw}\s+interface'],
        }
    
    def search_code(self, keywords: List[str], search_type: str = "all", max_results: int = 30) -> str:
        """Search with OR logic, ranked by frequency
        
        search_type: "all" (any match), "definitions" (class/func/var defs only)
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # Score files by keyword matches
        file_scores = defaultdict(lambda: {'matches': [], 'score': 0, 'keywords': set()})
        
        for file_path in self._get_code_files():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    matched_kw = set()
                    
                    for kw in keywords:
                        if search_type == "definitions":
                            # Check definition patterns
                            patterns = self.def_patterns.get(file_path.suffix, [])
                            for pattern in patterns:
                                if re.search(pattern.format(kw=re.escape(kw)), line, re.IGNORECASE):
                                    matched_kw.add(kw)
                                    break
                        else:
                            # Simple keyword match
                            if re.search(re.escape(kw), line, re.IGNORECASE):
                                matched_kw.add(kw)
                    
                    if matched_kw:
                        rel_path = str(file_path.relative_to(self.repo_path))
                        file_scores[rel_path]['matches'].append({
                            'line': line_num,
                            'text': line.strip(),
                            'keywords': matched_kw
                        })
                        file_scores[rel_path]['score'] += len(matched_kw)
                        file_scores[rel_path]['keywords'].update(matched_kw)
            except:
                continue
        
        if not file_scores:
            return f"No results for: {', '.join(keywords)}"
        
        # Sort by score
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Format results
        results = [f"Results for: {', '.join(keywords)}", "=" * 70]
        shown = 0
        
        for path, data in sorted_files:
            if shown >= max_results:
                break
            
            kw_str = ', '.join(data['keywords'])
            results.append(f"\n📁 {path} (score: {data['score']}, matched: {kw_str})")
            
            for match in data['matches'][:8]:  # Max 8 lines per file
                if shown >= max_results:
                    break
                results.append(f"   {match['line']:4d}: {match['text']}")
                shown += 1
            
            if len(data['matches']) > 8:
                results.append(f"   ... +{len(data['matches']) - 8} more")
        
        results.append("=" * 70)
        return "\n".join(results)
    
    def search_in_file(self, file_path: str, keywords: List[str], context_lines: int = 3) -> str:
        """Search within specific file with context"""
        if isinstance(keywords, str):
            keywords = [keywords]
        
        target = self.repo_path / file_path
        if not target.exists():
            return f"File not found: {file_path}"
        
        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except:
            return f"Error reading: {file_path}"
        
        results = [f"Search in {file_path} for: {', '.join(keywords)}", "=" * 70]
        
        for i, line in enumerate(lines):
            matched_kw = {kw for kw in keywords if re.search(re.escape(kw), line, re.IGNORECASE)}
            
            if matched_kw:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                results.append(f"\nMatch at line {i+1} [{', '.join(matched_kw)}]:")
                for j in range(start, end):
                    prefix = ">>>" if j == i else "   "
                    results.append(f"{prefix} {j+1:4d} | {lines[j].rstrip()}")
        
        if len(results) == 2:
            results.append("No matches found")
        
        results.append("=" * 70)
        return "\n".join(results)
    
    def get_file_outline(self, file_path: str) -> str:
        """Extract function/class definitions with line numbers"""
        target = self.repo_path / file_path
        if not target.exists():
            return f"File not found: {file_path}"
        
        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except:
            return f"Error reading: {file_path}"
        
        results = [f"Outline: {file_path}", "=" * 70]
        
        # Python patterns
        if target.suffix == '.py':
            for i, line in enumerate(lines):
                if line.strip().startswith('class '):
                    results.append(f"Line {i+1:4d}: {line.strip()}")
                elif line.strip().startswith('def '):
                    indent = len(line) - len(line.lstrip())
                    prefix = "  " * (indent // 4)
                    results.append(f"Line {i+1:4d}: {prefix}{line.strip()}")
        
        # JavaScript/TypeScript patterns
        elif target.suffix in {'.js', '.ts', '.jsx', '.tsx'}:
            for i, line in enumerate(lines):
                if re.search(r'(class|function|const\s+\w+\s*=\s*function)', line):
                    results.append(f"Line {i+1:4d}: {line.strip()}")
        
        if len(results) == 2:
            results.append("No definitions found")
        
        results.append("=" * 70)
        results.append(f"Total lines: {len(lines)}")
        return "\n".join(results)
    
    def view_file_lines(self, file_path: str, start: int, end: int) -> str:
        """View specific line range"""
        target = self.repo_path / file_path
        if not target.exists():
            return f"File not found: {file_path}"
        
        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except:
            return f"Error reading: {file_path}"
        
        start = max(1, start)
        end = min(len(lines), end)
        
        results = [f"File: {file_path} (lines {start}-{end} of {len(lines)})", "=" * 70]
        for i in range(start - 1, end):
            results.append(f"{i+1:4d} | {lines[i].rstrip()}")
        results.append("=" * 70)
        
        return "\n".join(results)
    
    def scroll_file(self, file_path: str, action: str = "start", chunk_size: int = 50) -> str:
        """Scroll through file in chunks"""
        state_key = f"file_{file_path}"
        target = self.repo_path / file_path
        
        if not target.exists():
            return f"File not found: {file_path}"
        
        # Initialize or get state
        if action == "start" or state_key not in self.scroll_states:
            try:
                with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                self.scroll_states[state_key] = {'lines': lines, 'pos': 0, 'chunk': chunk_size}
            except:
                return f"Error reading: {file_path}"
        
        state = self.scroll_states[state_key]
        
        # Navigate
        if action == "next":
            state['pos'] = min(state['pos'] + state['chunk'], len(state['lines']) - state['chunk'])
        elif action == "prev":
            state['pos'] = max(0, state['pos'] - state['chunk'])
        
        start = state['pos']
        end = min(start + state['chunk'], len(state['lines']))
        
        results = [f"File: {file_path} (lines {start+1}-{end} of {len(state['lines'])})", "=" * 70]
        for i in range(start, end):
            results.append(f"{i+1:4d} | {state['lines'][i].rstrip()}")
        results.append("=" * 70)
        
        if end < len(state['lines']):
            results.append("More below - use action='next'")
        if start > 0:
            results.append("More above - use action='prev'")
        
        return "\n".join(results)
    
    def list_directory(self, path: str = ".") -> str:
        """List files and folders"""
        target = self.repo_path / path
        if not target.exists() or not target.is_dir():
            return f"Not a directory: {path}"
        
        results = [f"Directory: {path}", "=" * 70]
        
        try:
            for item in sorted(target.iterdir()):
                if item.name.startswith('.') or item.name == '__pycache__':
                    continue
                
                rel = item.relative_to(self.repo_path)
                if item.is_dir():
                    results.append(f"[DIR]  {rel}/")
                else:
                    size = item.stat().st_size / 1024
                    results.append(f"[FILE] {rel} ({size:.1f} KB)")
        except:
            return f"Error reading: {path}"
        
        results.append("=" * 70)
        return "\n".join(results)
    
    def get_imports(self, file_path: str) -> str:
        """Get all imports from file"""
        target = self.repo_path / file_path
        if not target.exists():
            return f"File not found: {file_path}"
        
        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except:
            return f"Error reading: {file_path}"
        
        results = [f"Imports in {file_path}:", "=" * 70]
        ext = target.suffix
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Python
            if ext == '.py' and (line.startswith('import ') or line.startswith('from ')):
                results.append(f"{i+1:4d} | {line}")
            
            # JavaScript/TypeScript
            elif ext in {'.js', '.ts', '.jsx', '.tsx'} and 'import' in line:
                results.append(f"{i+1:4d} | {line}")
            
            # Java
            elif ext == '.java' and line.startswith('import '):
                results.append(f"{i+1:4d} | {line}")
            
            # C/C++
            elif ext in {'.c', '.cpp', '.h'} and line.startswith('#include'):
                results.append(f"{i+1:4d} | {line}")
        
        if len(results) == 2:
            results.append("No imports found")
        
        results.append("=" * 70)
        return "\n".join(results)
    
    def find_files_with_imports(self, module: str) -> str:
        """Find files importing a module"""
        results = [f"Files importing '{module}':", "=" * 70]
        
        for file_path in self._get_code_files():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Simple pattern matching
                if re.search(rf'\bimport\b.*\b{re.escape(module)}\b', content, re.IGNORECASE):
                    rel = file_path.relative_to(self.repo_path)
                    results.append(f"  {rel}")
            except:
                continue
        
        if len(results) == 2:
            results.append("No files found")
        
        results.append("=" * 70)
        return "\n".join(results)
    
    def _get_code_files(self) -> List[Path]:
        """Get all code files in repo"""
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            # Skip hidden and generated dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                      {'__pycache__', 'node_modules', 'venv', '.git', 'build', 'dist'}]
            
            for name in filenames:
                path = Path(root) / name
                if path.suffix in self.valid_extensions:
                    files.append(path)
        
        return files
