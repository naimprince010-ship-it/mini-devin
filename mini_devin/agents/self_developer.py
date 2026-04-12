"""
Self-Developer Agent for Plodder

This module implements a specialized agent that is aware of its own codebase
and is optimized for autonomous self-improvement, bug fixing, and feature
development on the Plodder repository itself.
"""

from typing import Any, Optional
from ..orchestrator.agent import Agent, SYSTEM_PROMPT as BASE_SYSTEM_PROMPT
from ..core.llm_client import LLMClient
from ..core.tool_interface import ToolRegistry

SELF_DEVELOPER_SYSTEM_PROMPT = """You are the Self-Developer mode of Plodder. You are an autonomous AI software engineer whose primary mission is to improve yourself.

## Your Context
You are currently running as a process within the Plodder repository. The files you see in the workspace ARE your own source code. 

## Your Mission
1. Identify bugs in your own implementation and fix them.
2. Develop new features to enhance your capabilities.
3. Refactor code to improve maintainability and performance.
4. Keep your documentation up to date.

## Special Instructions
- **Self-Awareness**: When you modify a file in `mini_devin/`, you are changing your own brain and body. Be very careful.
- **Bootstrapping**: If you change core orchestrator or API logic, you may need to trigger a system restart to apply the changes. 
- **Verification**: Always run the full test suite (`pytest`) before and after making changes to your core. 
- **Legacy Protection**: Do not break existing features while adding new ones.

## Your Workflow
1. AUDIT: Scan your codebase for issues or opportunities.
2. PLAN: Create a detailed plan for the self-improvement task.
3. EXECUTE: Apply changes to your source code.
4. VERIFY: Run tests and linting to ensure everything still works.
5. DEPLOY: (Simulated) Your changes will be active on the next restart.

You have access to all standard Plodder tools (terminal, editor, browser). Use them wisely to evolve."""

class SelfDeveloperAgent(Agent):
    """
    A specialized agent for autonomous self-improvement.
    
    Inherits from the base Agent but uses a self-aware system prompt
    and has optimized logic for working on its own repository.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        working_directory: Optional[str] = None,
        max_iterations: int = 100,  # Higher default for complex self-improvement
        verbose: bool = True,
        **kwargs: Any
    ):
        super().__init__(
            llm_client=llm_client,
            tool_registry=tool_registry,
            working_directory=working_directory,
            max_iterations=max_iterations,
            verbose=verbose,
            **kwargs
        )
        
        # Override the system prompt with the self-developer version
        self.llm.set_system_prompt(SELF_DEVELOPER_SYSTEM_PROMPT)
        
    async def run_self_audit(self) -> str:
        """
        Run a proactive audit of the codebase to find improvement tasks.
        """
        self._log("Starting self-audit...")
        # This could call a specialized tool or just use the LLM to analyze the file tree
        audit_task = "Analyze the current repository and identify 3 potential quality improvements or missing features."
        # For now, we return a prompt that can be used to start a session
        return audit_task

def create_self_developer_agent(
    llm_client: Optional[LLMClient] = None,
    tool_registry: Optional[ToolRegistry] = None,
    working_directory: Optional[str] = None,
    **kwargs: Any
) -> SelfDeveloperAgent:
    """Factory function to create a SelfDeveloperAgent."""
    return SelfDeveloperAgent(
        llm_client=llm_client,
        tool_registry=tool_registry,
        working_directory=working_directory,
        **kwargs
    )
