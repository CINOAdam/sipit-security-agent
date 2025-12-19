#!/usr/bin/env python3
"""
Demo script for the security agent.

Shows the agent's decision-making without requiring real targets.
Uses mock tool execution to demonstrate the workflow.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any

from .tools import ToolResult, SecurityTool, SECURITY_TOOLS
from .goals import Goal, GoalType, Target
from .agent import SecurityAgent


# =============================================================================
# Mock Tool Execution
# =============================================================================

# Simulated outputs for common scenarios
MOCK_OUTPUTS = {
    "nmap": {
        "default": """Starting Nmap 7.94 ( https://nmap.org )
Nmap scan report for 192.168.1.100
Host is up (0.001s latency).

PORT     STATE SERVICE     VERSION
21/tcp   open  ftp         vsftpd 2.3.4
22/tcp   open  ssh         OpenSSH 4.7p1
80/tcp   open  http        Apache httpd 2.2.8
139/tcp  open  netbios-ssn Samba 3.0.20
445/tcp  open  netbios-ssn Samba 3.0.20
3632/tcp open  distcc      distccd v1
6667/tcp open  irc         UnrealIRCd

Nmap done: 1 IP address (1 host up) scanned in 5.23 seconds
""",
    },
    "nikto": {
        "default": """- Nikto v2.1.6
---------------------------------------------------------------------------
+ Target IP:          192.168.1.100
+ Target Hostname:    192.168.1.100
+ Target Port:        80
+ Start Time:         2024-01-15 10:30:00
---------------------------------------------------------------------------
+ Server: Apache/2.2.8 (Ubuntu) PHP/5.2.4-2ubuntu5.10
+ Retrieved x-powered-by header: PHP/5.2.4-2ubuntu5.10
+ /phpinfo.php: PHP information page found.
+ /test/: Directory indexing found.
+ /admin/: Admin login page found.
+ OSVDB-3233: /icons/README: Apache default file found.
---------------------------------------------------------------------------
+ End Time:           2024-01-15 10:35:00
+ 1 host(s) tested
""",
    },
    "gobuster": {
        "default": """===============================================================
Gobuster v3.1.0
===============================================================
[+] Url:                     http://192.168.1.100
[+] Method:                  GET
[+] Threads:                 10
[+] Wordlist:                /usr/share/wordlists/dirb/common.txt
===============================================================
/admin                (Status: 301) [Size: 314]
/backup               (Status: 301) [Size: 315]
/cgi-bin              (Status: 403) [Size: 289]
/images               (Status: 301) [Size: 315]
/index.php            (Status: 200) [Size: 2048]
/phpMyAdmin           (Status: 301) [Size: 319]
/robots.txt           (Status: 200) [Size: 47]
/uploads              (Status: 301) [Size: 316]
===============================================================
""",
    },
    "searchsploit": {
        "vsftpd": """
------------------------------------------- ---------------------------------
 Exploit Title                             |  Path
------------------------------------------- ---------------------------------
vsftpd 2.3.4 - Backdoor Command Execution  | unix/remote/49757.py
vsftpd 2.3.4 - Backdoor Command Execution  | unix/remote/17491.rb
------------------------------------------- ---------------------------------
""",
        "samba": """
------------------------------------------- ---------------------------------
 Exploit Title                             |  Path
------------------------------------------- ---------------------------------
Samba 3.0.20 < 3.0.25rc3 - 'Username' map  | unix/remote/16320.rb
Samba 3.0.20 - Remote Heap Overflow        | linux/remote/7701.txt
------------------------------------------- ---------------------------------
""",
    },
    "curl": {
        "default": """<!DOCTYPE html>
<html>
<head><title>Welcome</title></head>
<body>
<h1>Welcome to the test server</h1>
<p>Login at <a href="/admin">admin panel</a></p>
</body>
</html>
""",
    },
    "nc": {
        "shell": """Connection received from 192.168.1.100
Linux metasploitable 2.6.24-16-server #1 SMP
uid=0(root) gid=0(root) groups=0(root)
$
""",
    },
}


class MockSecurityTool(SecurityTool):
    """A security tool with mocked execution for demo purposes."""

    def __init__(self, real_tool: SecurityTool):
        self.name = real_tool.name
        self.description = real_tool.description
        self.category = real_tool.category
        self.parameters = real_tool.parameters
        self.examples = real_tool.examples
        self._real_tool = real_tool

    def build_command(self, **kwargs) -> str:
        return self._real_tool.build_command(**kwargs)

    async def execute(self, timeout: int = 300, **kwargs) -> ToolResult:
        """Mock execution with simulated output."""
        command = self.build_command(**kwargs)

        # Get mock output
        tool_mocks = MOCK_OUTPUTS.get(self.name, {})

        # Try to find specific mock based on parameters
        output = None
        for key, value in kwargs.items():
            if isinstance(value, str):
                for mock_key, mock_output in tool_mocks.items():
                    if mock_key in value.lower():
                        output = mock_output
                        break
            if output:
                break

        if not output:
            output = tool_mocks.get("default", f"[Mock output for {self.name}]")

        # Simulate some delay
        await asyncio.sleep(0.5)

        return ToolResult(
            tool=self.name,
            command=command,
            success=True,
            output=output,
            parsed_data=self._real_tool.parse_output(output) if hasattr(self._real_tool, 'parse_output') else {},
        )


class MockSecurityAgent(SecurityAgent):
    """Security agent with mocked tool execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace tools with mocked versions
        self.tools = {
            name: MockSecurityTool(tool)
            for name, tool in SECURITY_TOOLS.items()
        }


# =============================================================================
# Demo Runner
# =============================================================================

async def run_demo(verbose: bool = True):
    """Run a demo of the security agent."""
    print("="*70)
    print("SECURITY AGENT DEMO")
    print("="*70)
    print("\nThis demo shows the agent's decision-making process")
    print("using mocked tool outputs (no real network activity).\n")

    # Create a demo goal
    goal = Goal(
        id="demo_shell",
        type=GoalType.GET_SHELL,
        description="Get a shell on the target system",
        target=Target(
            ip="192.168.1.100",
            hostname="demo-target",
            notes="Simulated vulnerable target",
        ),
        success_criteria="Establish shell access on the target",
        hints=[
            "Start with nmap to find open ports",
            "Look for vulnerable services",
            "Use searchsploit to find exploits",
        ],
        max_steps=10,
    )

    # Run with mock agent
    agent = MockSecurityAgent(model_name="gpt-4o-mini")

    print("Starting agent run...")
    print("-"*70)

    memory = await agent.run(goal, verbose=verbose)

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print(f"\nSteps taken: {len(memory.actions)}")
    print("\nAction sequence:")
    for i, action in enumerate(memory.actions):
        print(f"  {i+1}. {action['tool']}({list(action['parameters'].keys())})")

    return memory


async def run_decision_demo():
    """Demo showing just the decision-making without execution."""
    print("="*70)
    print("DECISION-MAKING DEMO")
    print("="*70)
    print("\nThis shows how the agent reasons about what tool to use.\n")

    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel
    from .agent import ToolDecision
    from .tools import format_tools_for_prompt

    model = OpenAIModel("gpt-4o-mini")
    tools_desc = format_tools_for_prompt()

    agent = Agent(
        model=model,
        result_type=ToolDecision,
        system_prompt=f"""You are an expert penetration tester.

{tools_desc}

Decide which tool to use next to achieve the goal.
""",
    )

    scenarios = [
        "I need to find what ports are open on 192.168.1.100",
        "I found vsftpd 2.3.4 running. I need to check for known exploits.",
        "There's a web server on port 80. I want to find hidden directories.",
        "I found an exploit for vsftpd 2.3.4 backdoor. Time to get a shell.",
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        print("-"*50)

        result = await agent.run(scenario)
        decision = result.data

        print(f"Reasoning: {decision.reasoning}")
        print(f"Tool: {decision.tool_name}")
        print(f"Parameters: {decision.parameters}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Security Agent Demo")
    parser.add_argument("--decisions", "-d", action="store_true",
                       help="Run decision-making demo only")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Less verbose output")

    args = parser.parse_args()

    if args.decisions:
        asyncio.run(run_decision_demo())
    else:
        asyncio.run(run_demo(verbose=not args.quiet))
