import asyncio
from openhands.sdk import LLM, Agent, Conversation, AgentContext, Skill
from pydantic import SecretStr

async def main():
    # 1. Load skill with MCP config
    skill = Skill.from_file(".openhands/skills/semantic-search.md")
    
    # 2. Create agent with MCP
    llm = LLM(
        model="claude-sonnet-4-5",
        api_key=SecretStr("your-key")
    )
    
    context = AgentContext(
        skills=[skill],
        mcp_servers_config_path=".openhands/mcp_servers.json"
    )
    
    agent = Agent(llm=llm, context=context)
    
    # 3. Run agent
    conversation = Conversation(agent=agent, workspace=".")
    conversation.send_message("Find validation code")
    await conversation.run()
    
    print(conversation.agent_final_response())

asyncio.run(main())
