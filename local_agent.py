#!/usr/bin/env python3
"""
Local model agent for self-improvement loop.

Uses the fine-tuned model directly for inference instead of external APIs.
"""

import json
import re
import torch
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .tools import get_all_tools, ToolResult
from .goals import Goal
from .verification import GoalVerifier, VerificationResult


# Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_ADAPTER = Path(__file__).parent / "multiskill_model" / "final"


@dataclass
class LocalAgentMemory:
    """Stores agent's action history."""
    goal: Goal
    actions: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    current_step: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_action(self, tool_name: str, parameters: Dict[str, Any], result: ToolResult):
        self.actions.append({
            "step": self.current_step,
            "tool": tool_name,
            "parameters": parameters,
            "success": result.success,
            "output_preview": result.output[:500] if result.output else "",
        })
        self.tool_results.append(result)
        self.current_step += 1

    def to_chain(self) -> List[Dict[str, Any]]:
        return [
            {"step": a["step"], "tool": a["tool"], "parameters": a["parameters"]}
            for a in self.actions
        ]


class LocalSecurityAgent:
    """
    Security agent powered by local fine-tuned model.

    Uses direct inference instead of API calls.
    """

    def __init__(
        self,
        adapter_path: Path = DEFAULT_ADAPTER,
        device: str = "auto",
        use_8bit: bool = True,
    ):
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.tools = get_all_tools()
        self.verifier = GoalVerifier()
        self.device = device
        self.use_8bit = use_8bit

    def load_model(self):
        """Load model and adapter (lazy loading)."""
        if self.model is not None:
            return

        print("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                load_in_8bit=True,
                device_map=self.device,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map=self.device,
            )

        if self.adapter_path.exists():
            print(f"Loading adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
        else:
            print("No adapter found, using base model")

        self.model.eval()

    def format_prompt(self, goal: Goal, memory: LocalAgentMemory) -> str:
        """Format prompt in EXACT training format."""
        # Determine skill from goal type/description
        skill = self._infer_skill(goal)

        system = f"""You are an expert penetration tester.
Skill type: {skill}
Goal: {goal.description}
Target: {goal.target.ip}

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON."""

        # First step - use exact training format
        if memory.current_step == 0:
            user_msg = f"Begin the {skill.replace('_', ' ')} task. What's your first action?"
        else:
            # Continuation - use training continuation format
            last_tool = memory.actions[-1]["tool"] if memory.actions else "previous tool"
            user_msg = f"Tool {last_tool} executed. Continue with {skill.replace('_', ' ')}."

        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"

    def _infer_skill(self, goal: Goal) -> str:
        """Infer skill type from goal."""
        desc = goal.description.lower()

        if "port" in desc or "service" in desc or "nmap" in desc:
            return "skill1_port_enum"
        elif "vulnerab" in desc or "nikto" in desc or "web scan" in desc:
            return "skill2_web_scan"
        elif "director" in desc or "gobuster" in desc or "hidden" in desc:
            return "skill3_dir_brute"
        elif "credential" in desc or "login" in desc or "password" in desc or "ssh" in desc or "ftp" in desc:
            return "skill4_default_creds"
        elif "api" in desc or "endpoint" in desc or "rest" in desc:
            return "skill5_api_enum"
        else:
            return "skill1_port_enum"  # Default

    def generate_decision(self, prompt: str) -> Dict[str, Any]:
        """Generate tool decision from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return self._parse_decision(response)

    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """Parse JSON decision from model response."""
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "tool_name": data.get("tool_name", "nmap"),
                    "parameters": data.get("parameters", {}),
                    "reasoning": data.get("reasoning", ""),
                    "is_complete": data.get("is_complete", False),
                }
            except json.JSONDecodeError:
                pass

        # Fallback: extract tool name from response
        tools = ["nmap", "nikto", "gobuster", "nc", "curl", "searchsploit"]
        for tool in tools:
            if tool in response.lower():
                return {
                    "tool_name": tool,
                    "parameters": {},  # Will be filled with defaults later
                    "reasoning": response[:100],
                    "is_complete": False,
                }

        # Last resort
        return {
            "tool_name": "nmap",
            "parameters": {},  # Will be filled with defaults later
            "reasoning": "Fallback to nmap",
            "is_complete": False,
        }

    def _default_params(self, tool: str, goal: Goal = None) -> Dict[str, Any]:
        """Get default parameters for a tool based on goal."""
        target = goal.target.ip if goal else "192.168.0.41"

        # Get port from goal's known services
        port = 80
        if goal and goal.target.known_services:
            port = goal.target.known_services[0].get("port", 80)

        defaults = {
            "nmap": {"target": target, "options": f"-sV -p {port}"},
            "nikto": {"target": target, "port": port},
            "gobuster": {"url": f"http://{target}:{port}", "wordlist": "common"},
            "nc": {"host": target, "port": port},
            "curl": {"url": f"http://{target}:{port}"},
            "searchsploit": {"query": "apache"},
        }
        return defaults.get(tool, {})

    async def run(self, goal: Goal, verbose: bool = True) -> LocalAgentMemory:
        """Run agent toward a goal using local model."""
        self.load_model()

        memory = LocalAgentMemory(goal=goal)
        memory.start_time = datetime.now()

        if verbose:
            print(f"\n{'='*60}")
            print(f"LOCAL AGENT: {goal.description}")
            print(f"Target: {goal.target.ip}")
            print(f"{'='*60}")

        while memory.current_step < goal.max_steps:
            prompt = self.format_prompt(goal, memory)

            if verbose:
                print(f"\n--- Step {memory.current_step + 1}/{goal.max_steps} ---")

            try:
                # Get decision from local model
                decision = self.generate_decision(prompt)

                if verbose:
                    print(f"Tool: {decision['tool_name']}")
                    print(f"Params: {decision['parameters']}")

                if decision["is_complete"]:
                    if verbose:
                        print("Agent believes task is complete")
                    break

                # Execute tool
                tool = self.tools.get(decision["tool_name"])
                if not tool:
                    if verbose:
                        print(f"Unknown tool: {decision['tool_name']}")
                    memory.add_action(
                        decision["tool_name"],
                        decision["parameters"],
                        ToolResult(tool=decision["tool_name"], command="", success=False, output="", error="Unknown tool"),
                    )
                    continue

                # Get parameters - use defaults if empty
                params = decision["parameters"].copy() if decision["parameters"] else {}
                if not params:
                    params = self._default_params(decision["tool_name"], goal)

                # Adjust parameters for goal target
                if "target" in params:
                    params["target"] = goal.target.ip
                if "url" in params and "192.168.0.41" in str(params["url"]):
                    params["url"] = params["url"].replace("192.168.0.41", goal.target.ip)
                if "host" in params:
                    params["host"] = goal.target.ip

                if verbose:
                    print(f"Executing {decision['tool_name']}...")

                result = await tool.execute(**params)

                if verbose:
                    status = "✓" if result.success else "✗"
                    print(f"Result: {status}")
                    if result.output:
                        print(f"Output: {result.output[:150]}...")

                memory.add_action(decision["tool_name"], params, result)

                # Check verification
                verification = self.verifier.verify(goal, memory.tool_results)
                if verification.result == VerificationResult.SUCCESS:
                    if verbose:
                        print(f"\n*** GOAL ACHIEVED ***")
                    break

            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                continue

        memory.end_time = datetime.now()

        if verbose:
            duration = (memory.end_time - memory.start_time).total_seconds()
            print(f"\nCompleted in {memory.current_step} steps ({duration:.1f}s)")

        return memory


async def test_local_agent():
    """Quick test of local agent."""
    from .goals import Goal, GoalType, Target

    goal = Goal(
        id="test_port_scan",
        type=GoalType.ENUMERATE,
        description="Enumerate open ports and services on target",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Identify open ports",
        hints=["Use nmap"],
        max_steps=3,
    )

    agent = LocalSecurityAgent()
    memory = await agent.run(goal, verbose=True)

    print(f"\nChain: {[a['tool'] for a in memory.actions]}")
    return memory


if __name__ == "__main__":
    asyncio.run(test_local_agent())
