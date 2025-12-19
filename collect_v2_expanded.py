#!/usr/bin/env python3
"""
Expanded V2 chain collection with harder goals.

Includes:
- More enumeration goals (for reproducibility)
- Exploitation goals (command injection, SQL injection)
- More runs per goal (5x)
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
from security_agent.goals import Goal, GoalType, Target
from security_agent.verification import GoalVerifier, VerificationResult
from security_agent.v2_trainer import ChainSimilarityScorer, ChainCollection


# =============================================================================
# EXPANDED GOALS - Mix of Easy, Medium, Hard
# =============================================================================

EXPANDED_GOALS = [
    # -------------------------------------------------------------------------
    # EASY: Basic Enumeration (should have high reproducibility)
    # -------------------------------------------------------------------------
    Goal(
        id="dvwa_basic_enum",
        type=GoalType.ENUMERATE,
        description="Perform basic enumeration of DVWA web application",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Identify web server, technologies, and potential entry points",
        hints=["Start with nmap", "Use nikto for web vulns"],
        max_steps=5,
    ),

    Goal(
        id="juice_basic_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate Juice Shop application structure",
        target=Target(ip="192.168.0.41", hostname="juice-shop",
                     known_services=[{"port": 8002, "service": "http"}]),
        success_criteria="Find API endpoints and application structure",
        hints=["Check for /api/ endpoints", "Look at page source"],
        max_steps=5,
    ),

    Goal(
        id="vampi_api_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate VAmPI REST API endpoints",
        target=Target(ip="192.168.0.41", hostname="vampi",
                     known_services=[{"port": 8008, "service": "http"}]),
        success_criteria="Discover API documentation and endpoints",
        hints=["Check /openapi.json", "Try common API paths"],
        max_steps=5,
    ),

    # -------------------------------------------------------------------------
    # MEDIUM: Deeper Enumeration with Specific Targets
    # -------------------------------------------------------------------------
    Goal(
        id="dvwa_vuln_scan",
        type=GoalType.ENUMERATE,
        description="Scan DVWA for specific vulnerabilities",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Identify SQL injection, XSS, and command injection points",
        hints=["Use nikto with tuning", "Check /vulnerabilities/ paths"],
        max_steps=8,
    ),

    Goal(
        id="bwapp_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate bWAPP vulnerable web application",
        target=Target(ip="192.168.0.41", hostname="bwapp",
                     known_services=[{"port": 8003, "service": "http"}]),
        success_criteria="Find login page and vulnerability categories",
        hints=["Check application structure", "Look for bug categories"],
        max_steps=6,
    ),

    Goal(
        id="mutillidae_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate Mutillidae OWASP application",
        target=Target(ip="192.168.0.41", hostname="mutillidae",
                     known_services=[{"port": 8006, "service": "http"}]),
        success_criteria="Map application pages and vulnerability types",
        hints=["Check menu structure", "Look for OWASP Top 10 pages"],
        max_steps=6,
    ),

    # -------------------------------------------------------------------------
    # HARD: Exploitation Goals
    # -------------------------------------------------------------------------
    Goal(
        id="dvwa_cmd_injection",
        type=GoalType.GET_SHELL,
        description="Exploit command injection vulnerability in DVWA to execute commands",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Execute arbitrary commands via the ping utility",
        hints=[
            "Target: /vulnerabilities/exec/",
            "Use curl to submit forms",
            "Chain commands with ; or |",
            "Try: 127.0.0.1; id",
        ],
        max_steps=10,
    ),

    Goal(
        id="dvwa_sqli_extract",
        type=GoalType.FIND_FLAG,
        description="Use SQL injection to extract data from DVWA database",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Extract user credentials from the database",
        hints=[
            "Target: /vulnerabilities/sqli/",
            "Use UNION-based injection",
            "Try: 1' UNION SELECT user,password FROM users--",
        ],
        max_steps=10,
    ),

    Goal(
        id="juice_sqli_admin",
        type=GoalType.CREDENTIAL,
        description="Use SQL injection to gain admin access to Juice Shop",
        target=Target(ip="192.168.0.41", hostname="juice-shop",
                     known_services=[{"port": 8002, "service": "http"}]),
        success_criteria="Login as admin user via SQL injection",
        hints=[
            "Target login form",
            "Admin email: admin@juice-sh.op",
            "Try: ' OR 1=1--",
        ],
        max_steps=10,
    ),

    Goal(
        id="vampi_api_exploit",
        type=GoalType.CREDENTIAL,
        description="Exploit VAmPI API vulnerabilities to access protected data",
        target=Target(ip="192.168.0.41", hostname="vampi",
                     known_services=[{"port": 8008, "service": "http"}]),
        success_criteria="Access unauthorized data or credentials via API flaws",
        hints=[
            "Check for IDOR vulnerabilities",
            "Try accessing other users' data",
            "Look for authentication bypass",
        ],
        max_steps=10,
    ),
]


async def collect_expanded_chains(
    runs_per_goal: int = 5,
    model: str = "openai:gpt-4o",
    output_dir: str = None,
):
    """Collect expanded chain dataset."""
    print("=" * 70)
    print("EXPANDED V2 CHAIN COLLECTION")
    print("=" * 70)
    print(f"Goals: {len(EXPANDED_GOALS)}")
    print(f"Runs per goal: {runs_per_goal}")
    print(f"Total expected runs: {len(EXPANDED_GOALS) * runs_per_goal}")
    print(f"Model: {model}")
    print("=" * 70)

    if output_dir is None:
        output_dir = Path(__file__).parent / "v2_chains_expanded"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    agent = SecurityAgent(model_name=model)
    verifier = GoalVerifier()
    similarity_scorer = ChainSimilarityScorer()

    all_chains = []
    collections = {}
    stats = {
        "easy": {"total": 0, "successful": 0},
        "medium": {"total": 0, "successful": 0},
        "hard": {"total": 0, "successful": 0},
    }

    # Categorize goals
    easy_goals = [g for g in EXPANDED_GOALS if g.max_steps <= 5]
    medium_goals = [g for g in EXPANDED_GOALS if 5 < g.max_steps <= 8]
    hard_goals = [g for g in EXPANDED_GOALS if g.max_steps > 8]

    for goal in EXPANDED_GOALS:
        # Determine difficulty
        if goal.max_steps <= 5:
            difficulty = "easy"
        elif goal.max_steps <= 8:
            difficulty = "medium"
        else:
            difficulty = "hard"

        print(f"\n{'─' * 70}")
        print(f"[{difficulty.upper()}] {goal.id}")
        print(f"Goal: {goal.description}")
        print(f"{'─' * 70}")

        collection = ChainCollection(goal_id=goal.id)

        for run_idx in range(runs_per_goal):
            print(f"\n  Run {run_idx + 1}/{runs_per_goal}...", end=" ", flush=True)
            stats[difficulty]["total"] += 1

            try:
                memory = await agent.run(goal, verbose=False)
                verification = verifier.verify(goal, memory.tool_results)
                chain = extract_training_chain(memory, verification)

                collection.chains.append(chain)
                if chain.success:
                    collection.successful_chains.append(chain)
                    stats[difficulty]["successful"] += 1

                status = "✓" if chain.success else "✗"
                tools = [a["tool"] for a in chain.actions]
                print(f"{status} {len(chain.actions)} steps: {' → '.join(tools[:4])}")

            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")

        collections[goal.id] = collection
        success_rate = len(collection.successful_chains) / len(collection.chains) if collection.chains else 0
        print(f"\n  Result: {len(collection.successful_chains)}/{len(collection.chains)} ({success_rate:.0%})")

        all_chains.extend(collection.chains)

    # Reproducibility analysis
    print(f"\n{'=' * 70}")
    print("REPRODUCIBILITY ANALYSIS")
    print("=" * 70)

    reproducible_chains = []

    for goal_id, collection in collections.items():
        if len(collection.successful_chains) < 2:
            continue

        print(f"\n{goal_id}:")
        for chain in collection.successful_chains:
            similar_count = sum(
                1 for other in collection.successful_chains
                if chain is not other and similarity_scorer.combined_similarity(chain, other) >= 0.5
            )

            if similar_count >= 1:
                reproducible_chains.append(chain)
                tools = [a["tool"] for a in chain.actions]
                print(f"  ✓ {' → '.join(tools[:4])} (sim: {similar_count})")

    # Summary
    print(f"\n{'=' * 70}")
    print("COLLECTION SUMMARY")
    print("=" * 70)

    total_chains = len(all_chains)
    successful = sum(1 for c in all_chains if c.success)
    reproducible = len(reproducible_chains)

    print(f"\nTotal chains: {total_chains}")
    print(f"Successful: {successful} ({successful/total_chains:.1%})")
    print(f"Reproducible: {reproducible}")

    print("\nBy difficulty:")
    for diff in ["easy", "medium", "hard"]:
        s = stats[diff]
        rate = s["successful"] / s["total"] if s["total"] > 0 else 0
        print(f"  {diff}: {s['successful']}/{s['total']} ({rate:.0%})")

    # Save
    all_path = output_dir / "all_chains.json"
    with open(all_path, "w") as f:
        json.dump([{
            "goal_id": c.goal_id,
            "goal_description": c.goal_description,
            "target_ip": c.target_ip,
            "actions": c.actions,
            "success": c.success,
            "duration_seconds": c.duration_seconds,
        } for c in all_chains], f, indent=2)

    repro_path = output_dir / "reproducible_chains.json"
    with open(repro_path, "w") as f:
        json.dump([{
            "goal_id": c.goal_id,
            "goal_description": c.goal_description,
            "target_ip": c.target_ip,
            "actions": c.actions,
            "success": c.success,
            "duration_seconds": c.duration_seconds,
        } for c in reproducible_chains], f, indent=2)

    print(f"\nSaved to: {output_dir}")

    return all_chains, reproducible_chains


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", "-r", type=int, default=5)
    parser.add_argument("--model", "-m", default="openai:gpt-4o")
    args = parser.parse_args()

    asyncio.run(collect_expanded_chains(
        runs_per_goal=args.runs,
        model=args.model,
    ))
