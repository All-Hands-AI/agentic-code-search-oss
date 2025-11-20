import asyncio
import os
from openhands.sdk import LLM, Agent, Conversation, AgentContext, Skill
from pydantic import SecretStr

async def main():
    # 1. Load skill with MCP config
    skill = Skill.from_file(".openhands/skills/semantic-search.md")

    # 2. Create agent with MCP
    llm = LLM(
        model="claude-sonnet-4-5",
        api_key=SecretStr(os.getenv("LLM_API_KEY", "your-key"))
    )

    # Configure MCP servers - semantic code search
    mcp_config = {
        "mcpServers": {
            "semantic-code-search": {
                "command": "uv",
                "args": ["run", "python", "src/mcp_server/semantic_search_server.py"],
                "env": {}
            }
        }
    }

    # Create agent context with skills
    context = AgentContext(skills=[skill])

    # Create agent with MCP config
    agent = Agent(llm=llm, agent_context=context, mcp_config=mcp_config)

    # 3. Run agent
    conversation = Conversation(agent=agent, workspace=".")
    conversation.send_message("Find validation code in the repository")
    await conversation.run()

    print(conversation.agent_final_response())

if __name__ == "__main__":
    asyncio.run(main())
