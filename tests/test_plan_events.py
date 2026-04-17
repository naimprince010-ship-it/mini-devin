import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
import json

from mini_devin.orchestrator.agent import Agent
from mini_devin.schemas.state import TaskState, TaskGoal, AgentPhase

class TestPlanEvents(unittest.IsolatedAsyncioTestCase):
    async def test_plan_emission(self):
        # Mock LLM Client
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()
        mock_llm.get_usage_stats.return_value = {"total_tokens": 100}
        mock_llm.config.model = "test-model"
        
        # Sequence of responses to simulate planning and step progression
        responses = [
            # 1. Provide a plan
            MagicMock(
                content="I will do the following:\n1. Create a file\n2. Modify the file",
                tool_calls=[],
                finish_reason="stop"
            ),
            # 2. Start first step
            MagicMock(
                content="starting step 1: create a file",
                tool_calls=[MagicMock(id="1", name="terminal", arguments={"command": "touch test.txt"})],
                finish_reason="tool_calls"
            ),
            # 3. Complete first step and move to second
            MagicMock(
                content="Moving to step 2: modify the file",
                tool_calls=[MagicMock(id="2", name="terminal", arguments={"command": "echo 'hello' > test.txt"})],
                finish_reason="tool_calls"
            ),
            # 4. Finish
            MagicMock(
                content="Task complete. I created and modified the file.",
                tool_calls=[],
                finish_reason="stop"
            )
        ]
        mock_llm.complete.side_effect = responses
        
        # Callbacks tracker
        events = []
        async def on_plan_created(steps): events.append(("plan_created", steps))
        async def on_step_started(idx, text): events.append(("step_started", idx, text))
        async def on_step_completed(idx, text): events.append(("step_completed", idx, text))
        
        agent = Agent(llm_client=mock_llm)
        agent.callbacks = {
            "on_plan_created": on_plan_created,
            "on_step_started": on_step_started,
            "on_step_completed": on_step_completed,
            "on_message": AsyncMock(),
            "on_phase_change": AsyncMock(),
            "on_iteration": AsyncMock(),
        }
        
        # Mock _execute_tool to return success
        agent._execute_tool = AsyncMock(return_value="Success")
        
        task = TaskState(
            task_id="test_task",
            goal=TaskGoal(description="Test planning", acceptance_criteria=[])
        )
        
        await agent.run(task)
        
        # Check events
        print(f"\nCaptured events: {events}")
        
        self.assertGreaterEqual(len(events), 4, f"Expected at least 4 events, got {len(events)}: {events}")
        self.assertEqual(events[0][0], "plan_created")
        self.assertEqual(events[1], ("step_started", 0, "Create a file"))
        self.assertEqual(events[2], ("step_completed", 0, "Create a file"))
        self.assertEqual(events[3], ("step_started", 1, "Modify the file"))
        
        print("\nTest passed! Events emitted in correct order.")

if __name__ == "__main__":
    unittest.main()
