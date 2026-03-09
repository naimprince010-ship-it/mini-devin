
import asyncio
import os
from dotenv import load_dotenv
from mini_devin.core.llm_client import create_llm_client
from mini_devin.orchestrator.agent import Agent
from mini_devin.schemas.state import TaskState, TaskGoal, TaskStatus

load_dotenv()

async def test_agent():
    print("Initializing LLM client...")
    llm_client = create_llm_client()
    
    print("Initializing Agent...")
    agent = Agent(llm_client=llm_client)
    
    task_state = TaskState(
        task_id="test-task",
        goal=TaskGoal(description="Say 'Hello'"),
        status=TaskStatus.PENDING
    )
    
    print("Running Agent...")
    try:
        # Create a mock callback for tokens
        async def on_token(token, is_token=False):
            if is_token:
                print(token, end="", flush=True)

        agent.callbacks["on_message"] = on_token
        
        result = await agent.run(task_state)
        print("\nAgent finished.")
        print(f"Status: {result.status}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent())
