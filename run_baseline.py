#!/usr/bin/env python3
"""Run baseline evaluation against live targets."""

import asyncio
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from security_agent.agent import SecurityAgent
from security_agent.goals import Goal, GoalType, Target
from security_agent.verification import GoalVerifier, VerificationResult


async def run_juice_shop_enum():
    """Run agent against Juice Shop for enumeration."""
    print("=" * 70)
    print("BASELINE: Juice Shop Enumeration")
    print("=" * 70)

    goal = Goal(
        id="juice_shop_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate the Juice Shop web application to find vulnerabilities",
        target=Target(
            ip="192.168.0.41",
            hostname="juice-shop",
            known_services=[{"port": 8002, "service": "http"}],
        ),
        success_criteria="Identify hidden endpoints, API routes, and potential vulnerabilities",
        hints=[
            "Start with basic recon (curl, nikto)",
            "Use gobuster to find hidden directories",
            "Look for API endpoints",
        ],
        max_steps=8,  # Keep it short for baseline test
    )

    agent = SecurityAgent(model_name="openai:gpt-4o")
    memory = await agent.run(goal, verbose=True)

    # Verify
    verifier = GoalVerifier()
    verification = verifier.verify(goal, memory.tool_results)

    print(f"\n{'=' * 70}")
    print("RESULT")
    print("=" * 70)
    print(f"Steps taken: {len(memory.actions)}")
    print(f"Verification: {verification.result.value}")
    print(f"Confidence: {verification.confidence:.1%}")
    print(f"Evidence: {verification.evidence}")

    return memory, verification


async def run_dvwa_recon():
    """Run agent against DVWA for basic recon."""
    print("=" * 70)
    print("BASELINE: DVWA Reconnaissance")
    print("=" * 70)

    goal = Goal(
        id="dvwa_recon",
        type=GoalType.ENUMERATE,
        description="Perform reconnaissance on DVWA to identify attack surface",
        target=Target(
            ip="192.168.0.41",
            hostname="dvwa",
            known_services=[{"port": 8001, "service": "http"}],
        ),
        success_criteria="Identify web technologies, hidden paths, and potential vulnerabilities",
        hints=[
            "Check the web application structure",
            "Look for admin panels and login pages",
            "Identify backend technologies",
        ],
        max_steps=8,
    )

    agent = SecurityAgent(model_name="openai:gpt-4o")
    memory = await agent.run(goal, verbose=True)

    verifier = GoalVerifier()
    verification = verifier.verify(goal, memory.tool_results)

    print(f"\n{'=' * 70}")
    print("RESULT")
    print("=" * 70)
    print(f"Steps taken: {len(memory.actions)}")
    print(f"Verification: {verification.result.value}")
    print(f"Confidence: {verification.confidence:.1%}")

    return memory, verification


async def main():
    """Run baseline tests."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["juice", "dvwa", "both"], default="juice")
    args = parser.parse_args()

    results = []

    if args.target in ["juice", "both"]:
        mem, ver = await run_juice_shop_enum()
        results.append(("Juice Shop", ver.result == VerificationResult.SUCCESS))

    if args.target in ["dvwa", "both"]:
        mem, ver = await run_dvwa_recon()
        results.append(("DVWA", ver.result == VerificationResult.SUCCESS))

    print(f"\n{'=' * 70}")
    print("BASELINE SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ PARTIAL/FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
