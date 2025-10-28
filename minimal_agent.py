import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import torch
from vllm import LLM, SamplingParams
from tools import CodebaseTools


class CodeLocalizationAgent:
    """Minimal agent for code localization using vLLM"""
    
    def __init__(self, model_name: str, repo_path: str):
        print(f"Loading {model_name}...")
        
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=8192,
            dtype="auto",
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        self.tools = CodebaseTools(repo_path)
        self.conversation = []
        self.max_turns = 25
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        # Tool definitions
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search codebase with OR logic (any keyword matches). Returns ranked results. Use multiple related keywords!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keywords to search (OR logic). Use synonyms!"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["all", "definitions"],
                                "description": "all: any match | definitions: class/func/var only",
                                "default": "all"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 30
                            }
                        },
                        "required": ["keywords"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_in_file",
                    "description": "Search within a specific file with context lines",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "context_lines": {
                                "type": "integer",
                                "default": 3
                            }
                        },
                        "required": ["file_path", "keywords"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_outline",
                    "description": "Get function/class definitions with line numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "view_file_lines",
                    "description": "View specific line range",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"}
                        },
                        "required": ["file_path", "start", "end"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "scroll_file",
                    "description": "Scroll through file in chunks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": ["start", "next", "prev"],
                                "default": "start"
                            },
                            "chunk_size": {
                                "type": "integer",
                                "default": 50
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and folders",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "default": "."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_imports",
                    "description": "Get all imports from file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_files_with_imports",
                    "description": "Find files importing a module",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module": {"type": "string"}
                        },
                        "required": ["module"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool call"""
        try:
            if name == "search_code":
                return self.tools.search_code(
                    args.get("keywords", []),
                    args.get("search_type", "all"),
                    args.get("max_results", 30)
                )
            elif name == "search_in_file":
                return self.tools.search_in_file(
                    args.get("file_path"),
                    args.get("keywords", []),
                    args.get("context_lines", 3)
                )
            elif name == "get_file_outline":
                return self.tools.get_file_outline(args.get("file_path"))
            elif name == "view_file_lines":
                return self.tools.view_file_lines(
                    args.get("file_path"),
                    args.get("start", 1),
                    args.get("end", 1)
                )
            elif name == "scroll_file":
                return self.tools.scroll_file(
                    args.get("file_path"),
                    args.get("action", "start"),
                    args.get("chunk_size", 50)
                )
            elif name == "list_directory":
                return self.tools.list_directory(args.get("path", "."))
            elif name == "get_imports":
                return self.tools.get_imports(args.get("file_path"))
            elif name == "find_files_with_imports":
                return self.tools.find_files_with_imports(args.get("module"))
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate(self) -> Tuple[str, Optional[List[Dict]]]:
        """Generate response"""
        prompt = self.tokenizer.apply_chat_template(
            self.conversation,
            tools=self.tools_schema,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        
        # Parse tool calls
        tool_calls = None
        text = response
        
        if "<tool_call>" in response:
            tool_calls = []
            parts = response.split("<tool_call>")
            text = parts[0].strip()
            
            for part in parts[1:]:
                if "</tool_call>" in part:
                    try:
                        tool_json = part.split("</tool_call>")[0].strip()
                        tool_calls.append(json.loads(tool_json))
                    except:
                        pass
        
        return text.replace("<|im_end|>", "").strip(), tool_calls
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run agent"""
        print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}\n")
        
        system = """You are a code localization assistant. Find ALL relevant code through systematic exploration.

STRATEGY:
1. EXPLORE STRUCTURE: Use list_directory, get_file_outline to understand organization
2. SEARCH BROADLY: Use search_code with MULTIPLE keyword lists (synonyms, related terms). 
3. NARROW DOWN: Use search_in_file, view_file_lines to examine specific files
4. VERIFY: Read actual code to confirm relevance and identify precise line numbers

Use other tools such as get_imports, find_files_with_imports when relevant. Be thorough and precise in identifying relevant code. Use multiple strategies.

When done, provide: ANSWER: file1.py:10-20; file2.py:45-60"""

        self.conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": query}
        ]
        
        results = {"query": query, "turns": [], "answer": None, "success": False}
        start_time = time.time()
        
        for turn in range(self.max_turns):
            print(f"\n{'─'*70}\nTurn {turn + 1}/{self.max_turns}\n{'─'*70}")
            
            turn_start = time.time()
            text, tool_calls = self._generate()
            turn_time = time.time() - turn_start
            
            if text:
                print(f"💭 ({turn_time:.2f}s): {text}\n")
            
            turn_data = {"turn": turn + 1, "text": text, "tools": [], "time": turn_time}
            
            # Check for answer
            if "ANSWER:" in text:
                answer = [l for l in text.split('\n') if 'ANSWER:' in l]
                if answer:
                    results["answer"] = answer[0].split("ANSWER:")[1].strip()
                    results["success"] = True
                    results["turns"].append(turn_data)
                    print(f"\n{'='*70}\n✓ ANSWER: {results['answer']}\n{'='*70}")
                    break
            
            # Execute tools
            if tool_calls:
                self.conversation.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls
                })
                
                for call in tool_calls:
                    name = call.get("name")
                    args = call.get("arguments", {})
                    
                    print(f"🔧 {name}({json.dumps(args)})")
                    
                    result = self._execute_tool(name, args)
                    turn_data["tools"].append({"name": name, "args": args, "result": result})
                    
                    # Show result preview
                    lines = result.split('\n')
                    for line in lines:
                        print(f"   {line}")
                    print()
                    
                    self.conversation.append({
                        "role": "tool",
                        "name": name,
                        "content": result
                    })
            else:
                self.conversation.append({"role": "assistant", "content": text})
                if "ANSWER:" not in text:
                    self.conversation.append({
                        "role": "user",
                        "content": "Continue or provide final answer."
                    })
            
            results["turns"].append(turn_data)
        
        results["time"] = time.time() - start_time
        
        # Summary
        print("\nSUMMARY\n")
        print(f"Turns: {len(results['turns'])}")
        print(f"Time: {results['time']:.2f}s")
        print(f"Success: {results['success']}")
        if results['answer']:
            print(f"Answer: {results['answer']}")
        
        return results


def main():
    MODEL = "Qwen/Qwen3-4B"
    REPO = "./OpenHands"
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    agent = CodeLocalizationAgent(MODEL, REPO)
    
    query = "Find all relevant code to handling user messages"
    
    results = agent.run(query)
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results.json")


if __name__ == "__main__":
    main()
