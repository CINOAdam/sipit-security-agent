"""
Security Agent with PydanticAI.

The core agent loop that:
1. Receives a goal
2. Decides which tool to use
3. Executes the tool
4. Observes the result
5. Repeats until goal is achieved or max steps reached
"""

import json
import asyncio
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Load .env
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from pydantic import BaseModel
from pydantic_ai import Agent

from .tools import SecurityTool, ToolResult, get_tool, get_all_tools, format_tools_for_prompt
from .goals import Goal, GoalType
from .verification import GoalVerifier, GoalVerification, VerificationResult


# =============================================================================
# Agent State and Memory
# =============================================================================

@dataclass
class AgentMemory:
    """Stores the agent's action history and observations."""
    goal: Goal
    actions: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    current_step: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_action(self, tool_name: str, parameters: Dict[str, Any], result: ToolResult):
        """Record an action and its result."""
        self.actions.append({
            "step": self.current_step,
            "tool": tool_name,
            "parameters": parameters,
            "success": result.success,
            "output_preview": result.output[:500] if result.output else "",
        })
        self.tool_results.append(result)
        self.current_step += 1

    def get_history_prompt(self) -> str:
        """Format history for inclusion in prompt."""
        if not self.actions:
            return "No actions taken yet."

        lines = ["# Previous Actions:"]
        for action in self.actions[-10:]:  # Last 10 actions
            status = "SUCCESS" if action["success"] else "FAILED"
            lines.append(f"\n## Step {action['step']}: {action['tool']} [{status}]")
            lines.append(f"Parameters: {action['parameters']}")
            lines.append(f"Output preview: {action['output_preview'][:200]}...")

        return "\n".join(lines)

    def to_chain(self) -> List[Dict[str, Any]]:
        """Export as a training chain."""
        return [
            {
                "step": a["step"],
                "tool": a["tool"],
                "parameters": a["parameters"],
                "success": a["success"],
            }
            for a in self.actions
        ]


# =============================================================================
# Agent Decision Model
# =============================================================================

class ToolDecision(BaseModel):
    """The agent's decision about which tool to use next."""
    reasoning: str
    tool_name: str
    parameters: Dict[str, Any]
    is_complete: bool = False
    completion_reason: Optional[str] = None


# =============================================================================
# Security Agent
# =============================================================================

class SecurityAgent:
    """
    LLM-powered security agent that works toward pentesting goals.

    Uses PydanticAI for structured decision making.
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-4o",  # Use OpenAI gpt-4o by default
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name

        # Create the decision agent
        self.agent = Agent(
            model=model_name,
            output_type=ToolDecision,
            retries=3,
            system_prompt=self._build_system_prompt(),
        )

        self.verifier = GoalVerifier()
        self.tools = get_all_tools()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools_desc = format_tools_for_prompt()

        return f"""You are an expert penetration tester working toward a specific goal.

{tools_desc}

Your job is to:
1. Analyze the current situation based on previous actions and results
2. Decide which tool to use next
3. Provide the correct parameters for that tool
4. Continue until the goal is achieved

Guidelines:
- Start with reconnaissance (nmap, nikto, curl) to understand the target
- Use the information gathered to make informed decisions
- If something doesn't work, try alternative approaches
- Be systematic and thorough
- When you believe the goal is achieved, set is_complete=True

Your response MUST be a JSON object with these fields:
- reasoning: string explaining your choice
- tool_name: one of [nmap, nikto, gobuster, nc, curl, searchsploit]
- parameters: object with the tool parameters
- is_complete: boolean (false unless goal achieved)
"""

    async def run(self, goal: Goal, verbose: bool = True) -> AgentMemory:
        """
        Run the agent toward a goal.

        Returns the complete memory/history of actions taken.
        """
        memory = AgentMemory(goal=goal)
        memory.start_time = datetime.now()

        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting agent for goal: {goal.description}")
            print(f"Target: {goal.target.ip}")
            print(f"{'='*70}\n")

        while memory.current_step < goal.max_steps:
            # Build the prompt for this step
            prompt = self._build_step_prompt(goal, memory)

            if verbose:
                print(f"\n--- Step {memory.current_step + 1}/{goal.max_steps} ---")

            try:
                # Get decision from LLM
                result = await self.agent.run(prompt)
                decision = result.output

                if verbose:
                    print(f"Reasoning: {decision.reasoning[:100]}...")
                    print(f"Tool: {decision.tool_name}")
                    print(f"Parameters: {decision.parameters}")

                # Check if agent thinks it's done
                if decision.is_complete:
                    if verbose:
                        print(f"\nAgent believes goal is complete: {decision.completion_reason}")
                    break

                # Execute the tool
                tool = self.tools.get(decision.tool_name)
                if not tool:
                    if verbose:
                        print(f"Unknown tool: {decision.tool_name}")
                    # Record failed action
                    memory.add_action(
                        decision.tool_name,
                        decision.parameters,
                        ToolResult(
                            tool=decision.tool_name,
                            command="",
                            success=False,
                            output="",
                            error=f"Unknown tool: {decision.tool_name}",
                        ),
                    )
                    continue

                # Execute
                if verbose:
                    print(f"Executing: {decision.tool_name}...")

                tool_result = await tool.execute(**decision.parameters)

                if verbose:
                    status = "SUCCESS" if tool_result.success else "FAILED"
                    print(f"Result: {status}")
                    if tool_result.output:
                        print(f"Output preview: {tool_result.output[:200]}...")
                    if tool_result.error:
                        print(f"Error: {tool_result.error[:200]}")

                # Record action
                memory.add_action(decision.tool_name, decision.parameters, tool_result)

                # Check if goal is achieved
                verification = self.verifier.verify(goal, memory.tool_results)
                if verification.result == VerificationResult.SUCCESS:
                    if verbose:
                        print(f"\n*** GOAL ACHIEVED! ***")
                        print(f"Evidence: {verification.evidence}")
                    break

            except Exception as e:
                if verbose:
                    print(f"Error in step: {e}")
                # Continue to next step
                continue

        memory.end_time = datetime.now()

        if verbose:
            print(f"\n{'='*70}")
            print(f"Agent finished after {memory.current_step} steps")
            duration = (memory.end_time - memory.start_time).total_seconds()
            print(f"Duration: {duration:.1f}s")
            print(f"{'='*70}")

        return memory

    def _build_step_prompt(self, goal: Goal, memory: AgentMemory) -> str:
        """Build the prompt for the current step."""
        return f"""
{goal.to_prompt()}

{memory.get_history_prompt()}

Based on the goal and previous actions, decide what tool to use next.
If you believe the goal has been achieved, set is_complete=True.
"""


# =============================================================================
# Training Data Extraction
# =============================================================================

@dataclass
class TrainingChain:
    """A chain of actions suitable for training."""
    goal_id: str
    goal_description: str
    target_ip: str
    actions: List[Dict[str, Any]]
    success: bool
    verification: GoalVerification
    duration_seconds: float


def extract_training_chain(memory: AgentMemory, verification: GoalVerification) -> TrainingChain:
    """Extract a training chain from agent memory."""
    duration = 0.0
    if memory.start_time and memory.end_time:
        duration = (memory.end_time - memory.start_time).total_seconds()

    return TrainingChain(
        goal_id=memory.goal.id,
        goal_description=memory.goal.description,
        target_ip=memory.goal.target.ip,
        actions=memory.to_chain(),
        success=verification.result == VerificationResult.SUCCESS,
        verification=verification,
        duration_seconds=duration,
    )


# =============================================================================
# CLI Interface
# =============================================================================

async def run_agent_cli(
    goal_id: str,
    target_ip: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
):
    """Run the agent from CLI."""
    from .goals import get_goal, create_basic_shell_goal

    # Get or create goal
    if goal_id in ["shell", "get_shell"]:
        if not target_ip:
            raise ValueError("target_ip required for shell goal")
        goal = create_basic_shell_goal(target_ip)
    else:
        goal = get_goal(goal_id)
        if not goal:
            raise ValueError(f"Unknown goal: {goal_id}")

        # Override target IP if provided
        if target_ip:
            goal.target.ip = target_ip

    # Create and run agent
    agent = SecurityAgent(model_name=model)
    memory = await agent.run(goal, verbose=verbose)

    # Verify and extract training chain
    verifier = GoalVerifier()
    verification = verifier.verify(goal, memory.tool_results)
    chain = extract_training_chain(memory, verification)

    return chain


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run security agent")
    parser.add_argument("goal", help="Goal ID or 'shell' for basic shell goal")
    parser.add_argument("--target", "-t", help="Target IP address")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    chain = asyncio.run(run_agent_cli(
        args.goal,
        target_ip=args.target,
        model=args.model,
        verbose=not args.quiet,
    ))

    print(f"\nResult: {'SUCCESS' if chain.success else 'FAILURE'}")
    print(f"Steps: {len(chain.actions)}")
    print(f"Duration: {chain.duration_seconds:.1f}s")
