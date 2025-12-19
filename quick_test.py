#!/usr/bin/env python3
"""Quick test of security agent decision-making."""

import asyncio
import sys
from pathlib import Path

# Load .env from parent directory
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from security_agent.tools import format_tools_for_prompt
from security_agent.goals import Goal, GoalType, Target

from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Dict, Any, Optional


class ToolDecision(BaseModel):
    """The agent's decision about which tool to use next."""
    reasoning: str
    tool_name: str
    parameters: Dict[str, Any]
    is_complete: bool = False


async def test_decision_making():
    """Test the agent's tool selection logic."""
    print("=" * 70)
    print("SECURITY AGENT - DECISION MAKING TEST")
    print("=" * 70)

    tools_desc = format_tools_for_prompt()

    # Try Anthropic first (Claude Code likely has this key), fall back to OpenAI
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = "anthropic:claude-sonnet-4-20250514"
    else:
        model = "openai:gpt-4o"

    print(f"Using model: {model}\n")

    agent = Agent(
        model=model,
        output_type=ToolDecision,
        retries=3,
        system_prompt=f"""You are an expert penetration tester. You must respond with a JSON object containing your tool decision.

{tools_desc}

Given a goal and situation, decide which tool to use next.

Your response MUST be a JSON object with these fields:
- reasoning: string explaining your choice
- tool_name: one of [nmap, nikto, gobuster, nc, curl, searchsploit]
- parameters: object with the tool parameters
- is_complete: boolean (false unless goal achieved)
""",
    )

    # Test scenarios
    scenarios = [
        {
            "name": "Initial Recon",
            "prompt": """Goal: Find vulnerabilities in DVWA web application
Target: http://192.168.0.41:8001
Status: Just starting, no actions taken yet.

What tool should I use first?""",
        },
        {
            "name": "After Port Scan",
            "prompt": """Goal: Get shell access on target
Target: 10.10.10.183

Previous actions:
- nmap scan found: port 21 (vsftpd 2.3.4), port 22 (SSH), port 80 (Apache)

The vsftpd version 2.3.4 is interesting. What should I do next?""",
        },
        {
            "name": "Web Enumeration",
            "prompt": """Goal: Find hidden endpoints on web application
Target: http://192.168.0.41:8002 (Juice Shop)

Previous actions:
- curl showed it's a Node.js application with Angular frontend

What tool should I use to find hidden directories or API endpoints?""",
        },
        {
            "name": "Exploit Research",
            "prompt": """Goal: Exploit vsftpd 2.3.4
Target: 10.10.10.183

I found vsftpd 2.3.4 running on port 21. I need to check if there are known exploits.

What tool and query should I use?""",
        },
    ]

    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'─' * 70}")
        print(f"Prompt: {scenario['prompt'][:100]}...")

        try:
            result = await agent.run(scenario["prompt"])
            decision = result.output

            print(f"\n✓ Decision:")
            print(f"  Tool: {decision.tool_name}")
            print(f"  Parameters: {decision.parameters}")
            print(f"  Reasoning: {decision.reasoning[:150]}...")

        except Exception as e:
            print(f"\n✗ Error: {e}")

    print(f"\n{'=' * 70}")
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_decision_making())
