"""
Example: Using semantic search with a local vLLM-served model.

This demonstrates how to integrate semantic code search with a locally
served LLM through vLLM, avoiding external API dependencies.
"""

import asyncio
import os
from openhands.sdk import LLM, Agent, Conversation, AgentContext, Skill
from pydantic import SecretStr

async def main():
    # 1. Load semantic search skill
    skill = Skill.from_file(".openhands/skills/semantic-search.md")

    # 2. Configure LLM to use local vLLM server
    # vLLM should be running at http://localhost:8000
    # Start it with: vllm serve your-model-name --host 0.0.0.0 --port 8000
    llm = LLM(
        # Model name as configured in vLLM
        model="deepseek-coder-33b-instruct",  # or whatever model you're serving

        # vLLM OpenAI-compatible endpoint
        base_url="http://localhost:8000/v1",

        # vLLM doesn't require a real API key
        api_key=SecretStr("EMPTY"),

        # Optional: Custom provider to help litellm route correctly
        custom_llm_provider="openai",  # vLLM is OpenAI-compatible

        # Optional: Generation parameters
        temperature=0.0,
        max_output_tokens=4096,

        # Optional: Timeout for local server (can be longer than API calls)
        timeout=120,

        # Usage tracking identifier
        usage_id="vllm-agent"
    )

    # 3. Configure MCP server for semantic search
    mcp_config = {
        "mcpServers": {
            "semantic-code-search": {
                "command": "uv",
                "args": ["run", "python", "src/mcp_server/semantic_search_server.py"],
                "env": {}
            }
        }
    }

    # 4. Create agent context with skills
    context = AgentContext(skills=[skill])

    # 5. Create agent with local LLM and MCP tools
    agent = Agent(llm=llm, agent_context=context, mcp_config=mcp_config)

    # 6. Run conversation
    conversation = Conversation(agent=agent, workspace=".")

    print("=" * 80)
    print("Starting conversation with local vLLM model...")
    print("=" * 80)

    conversation.send_message(
        "Find code related to reward calculation and metrics in this repository"
    )
    await conversation.run()

    print("\n" + "=" * 80)
    print("Agent Response:")
    print("=" * 80)
    print(conversation.agent_final_response())

    # 7. Show metrics
    print("\n" + "=" * 80)
    print("LLM Metrics:")
    print("=" * 80)
    print(f"Total input tokens: {llm.metrics.accumulated_input_tokens}")
    print(f"Total output tokens: {llm.metrics.accumulated_output_tokens}")
    print(f"Total cost: ${llm.metrics.accumulated_cost:.4f}")  # Should be $0 for local

if __name__ == "__main__":
    asyncio.run(main())
