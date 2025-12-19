#!/usr/bin/env python3
"""
Collect V2 training chains from the security agent.

Runs multiple chains per goal and filters for reproducible, successful ones.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Load .env
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from security_agent.agent import SecurityAgent, AgentMemory, extract_training_chain
from security_agent.goals import Goal, GoalType, Target, GOAL_LIBRARY, HomelabTargets
from security_agent.verification import GoalVerifier, VerificationResult
from security_agent.v2_trainer import ChainSimilarityScorer, ChainCollection


# Define focused goals for V2 collection
V2_GOALS = [
    # Enumeration goal - should have reproducible patterns
    Goal(
        id="juice_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate Juice Shop to find API endpoints and vulnerabilities",
        target=Target(
            ip="192.168.0.41",
            hostname="juice-shop",
            known_services=[{"port": 8002, "service": "http"}],
        ),
        success_criteria="Find API endpoints, hidden paths, or vulnerabilities",
        hints=["Use nmap scripts", "Try curl to explore API", "Check /api/ paths"],
        max_steps=6,
    ),
    # DVWA recon
    Goal(
        id="dvwa_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate DVWA to identify all attack surfaces",
        target=Target(
            ip="192.168.0.41",
            hostname="dvwa",
            known_services=[{"port": 8001, "service": "http"}],
        ),
        success_criteria="Identify login pages, vulnerabilities, and backend tech",
        hints=["Scan with nmap", "Use nikto for web vulns", "Check for common paths"],
        max_steps=6,
    ),
    # VAmPI API enumeration
    Goal(
        id="vampi_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate VAmPI REST API to find endpoints",
        target=Target(
            ip="192.168.0.41",
            hostname="vampi",
            known_services=[{"port": 8008, "service": "http"}],
        ),
        success_criteria="Discover API endpoints and potential vulnerabilities",
        hints=["curl the API", "Look for OpenAPI/Swagger docs", "Try common API paths"],
        max_steps=6,
    ),
]


async def collect_chains(
    goals: list,
    runs_per_goal: int = 3,
    model: str = "openai:gpt-4o",
    output_dir: str = None,
):
    """Collect chains for V2 training."""
    print("=" * 70)
    print("V2 CHAIN COLLECTION")
    print("=" * 70)
    print(f"Goals: {len(goals)}")
    print(f"Runs per goal: {runs_per_goal}")
    print(f"Model: {model}")
    print("=" * 70)

    if output_dir is None:
        output_dir = Path(__file__).parent / "v2_chains"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    agent = SecurityAgent(model_name=model)
    verifier = GoalVerifier()
    similarity_scorer = ChainSimilarityScorer()

    all_chains = []
    collections = {}

    for goal in goals:
        print(f"\n{'─' * 70}")
        print(f"Goal: {goal.id} - {goal.description}")
        print(f"{'─' * 70}")

        collection = ChainCollection(goal_id=goal.id)

        for run_idx in range(runs_per_goal):
            print(f"\n  Run {run_idx + 1}/{runs_per_goal}...")

            try:
                memory = await agent.run(goal, verbose=False)
                verification = verifier.verify(goal, memory.tool_results)
                chain = extract_training_chain(memory, verification)

                collection.chains.append(chain)
                if chain.success:
                    collection.successful_chains.append(chain)

                status = "✓ SUCCESS" if chain.success else "✗ FAILED"
                print(f"    {status} - {len(chain.actions)} steps, {verification.confidence:.0%} confidence")

                # Show action sequence
                tools_used = [a["tool"] for a in chain.actions]
                print(f"    Tools: {' → '.join(tools_used)}")

            except Exception as e:
                print(f"    ✗ Error: {e}")

        collections[goal.id] = collection
        print(f"\n  Summary: {len(collection.successful_chains)}/{len(collection.chains)} successful")

    # Analyze reproducibility
    print(f"\n{'=' * 70}")
    print("REPRODUCIBILITY ANALYSIS")
    print("=" * 70)

    reproducible_chains = []

    for goal_id, collection in collections.items():
        print(f"\n{goal_id}:")

        if len(collection.successful_chains) < 2:
            print("  Not enough successful chains for reproducibility check")
            continue

        # Check similarity between successful chains
        for i, chain1 in enumerate(collection.successful_chains):
            similar_count = 0
            for j, chain2 in enumerate(collection.successful_chains):
                if i != j:
                    sim = similarity_scorer.combined_similarity(chain1, chain2)
                    if sim >= 0.5:
                        similar_count += 1

            if similar_count >= 1:  # At least one similar chain
                reproducible_chains.append(chain1)
                tools = [a["tool"] for a in chain1.actions]
                print(f"  ✓ Reproducible chain: {' → '.join(tools)} (sim with {similar_count} others)")

        all_chains.extend(collection.chains)

    # Save results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Total chains collected: {len(all_chains)}")
    print(f"Successful chains: {sum(1 for c in all_chains if c.success)}")
    print(f"Reproducible chains: {len(reproducible_chains)}")

    # Save all chains
    chains_data = []
    for chain in all_chains:
        chains_data.append({
            "goal_id": chain.goal_id,
            "goal_description": chain.goal_description,
            "target_ip": chain.target_ip,
            "actions": chain.actions,
            "success": chain.success,
            "duration_seconds": chain.duration_seconds,
        })

    all_chains_path = output_dir / "all_chains.json"
    with open(all_chains_path, "w") as f:
        json.dump(chains_data, f, indent=2)
    print(f"\nAll chains saved to: {all_chains_path}")

    # Save reproducible chains
    repro_data = []
    for chain in reproducible_chains:
        repro_data.append({
            "goal_id": chain.goal_id,
            "goal_description": chain.goal_description,
            "target_ip": chain.target_ip,
            "actions": chain.actions,
            "success": chain.success,
            "duration_seconds": chain.duration_seconds,
        })

    repro_path = output_dir / "reproducible_chains.json"
    with open(repro_path, "w") as f:
        json.dump(repro_data, f, indent=2)
    print(f"Reproducible chains saved to: {repro_path}")

    return all_chains, reproducible_chains


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", "-r", type=int, default=3, help="Runs per goal")
    parser.add_argument("--model", "-m", default="openai:gpt-4o", help="Model to use")
    args = parser.parse_args()

    await collect_chains(
        goals=V2_GOALS,
        runs_per_goal=args.runs,
        model=args.model,
    )


if __name__ == "__main__":
    asyncio.run(main())
