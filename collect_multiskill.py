#!/usr/bin/env python3
"""
Multi-skill chain collection for interpretability research.

Collects diverse chains across 5 skill types:
1. Port/Service Enumeration
2. Web Vulnerability Scanning
3. Directory Bruteforce
4. Default Credential Testing
5. API Enumeration

Goal: ~50 chains with clear skill differentiation for mechanistic analysis.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from security_agent.agent import SecurityAgent, AgentMemory, extract_training_chain
from security_agent.goals import Goal, GoalType, Target
from security_agent.verification import GoalVerifier, VerificationResult
from security_agent.v2_trainer import ChainSimilarityScorer


# =============================================================================
# SKILL 1: Port/Service Enumeration
# =============================================================================
SKILL1_PORT_ENUM = [
    Goal(
        id="skill1_dvwa_ports",
        type=GoalType.ENUMERATE,
        description="Enumerate open ports and services on DVWA server",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Identify open ports and service versions",
        hints=["Use nmap with version detection", "Target port 8001"],
        max_steps=3,
    ),
    Goal(
        id="skill1_juice_ports",
        type=GoalType.ENUMERATE,
        description="Enumerate open ports and services on Juice Shop",
        target=Target(ip="192.168.0.41", hostname="juice-shop",
                     known_services=[{"port": 8002, "service": "http"}]),
        success_criteria="Identify open ports and service versions",
        hints=["Use nmap with version detection", "Target port 8002"],
        max_steps=3,
    ),
    Goal(
        id="skill1_bwapp_ports",
        type=GoalType.ENUMERATE,
        description="Enumerate open ports and services on bWAPP",
        target=Target(ip="192.168.0.41", hostname="bwapp",
                     known_services=[{"port": 8003, "service": "http"}]),
        success_criteria="Identify open ports and service versions",
        hints=["Use nmap with version detection", "Target port 8003"],
        max_steps=3,
    ),
    Goal(
        id="skill1_msf_ports",
        type=GoalType.ENUMERATE,
        description="Enumerate open ports on Metasploitable2 FTP/SSH",
        target=Target(ip="192.168.0.41", hostname="metasploitable",
                     known_services=[{"port": 8021, "service": "ftp"}, {"port": 8022, "service": "ssh"}]),
        success_criteria="Identify FTP and SSH service versions",
        hints=["Use nmap", "Target ports 8021,8022"],
        max_steps=3,
    ),
]

# =============================================================================
# SKILL 2: Web Vulnerability Scanning
# =============================================================================
SKILL2_WEB_SCAN = [
    Goal(
        id="skill2_dvwa_webscan",
        type=GoalType.ENUMERATE,
        description="Scan DVWA for web vulnerabilities using nikto",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Identify web vulnerabilities and misconfigurations",
        hints=["Use nikto web scanner", "Target http://192.168.0.41:8001"],
        max_steps=3,
    ),
    Goal(
        id="skill2_juice_webscan",
        type=GoalType.ENUMERATE,
        description="Scan Juice Shop for web vulnerabilities using nikto",
        target=Target(ip="192.168.0.41", hostname="juice-shop",
                     known_services=[{"port": 8002, "service": "http"}]),
        success_criteria="Identify web vulnerabilities and misconfigurations",
        hints=["Use nikto web scanner", "Target http://192.168.0.41:8002"],
        max_steps=3,
    ),
    Goal(
        id="skill2_bwapp_webscan",
        type=GoalType.ENUMERATE,
        description="Scan bWAPP for web vulnerabilities using nikto",
        target=Target(ip="192.168.0.41", hostname="bwapp",
                     known_services=[{"port": 8003, "service": "http"}]),
        success_criteria="Identify web vulnerabilities and misconfigurations",
        hints=["Use nikto web scanner", "Target http://192.168.0.41:8003"],
        max_steps=3,
    ),
    Goal(
        id="skill2_mutillidae_webscan",
        type=GoalType.ENUMERATE,
        description="Scan Mutillidae for web vulnerabilities using nikto",
        target=Target(ip="192.168.0.41", hostname="mutillidae",
                     known_services=[{"port": 8006, "service": "http"}]),
        success_criteria="Identify web vulnerabilities and misconfigurations",
        hints=["Use nikto web scanner", "Target http://192.168.0.41:8006"],
        max_steps=3,
    ),
]

# =============================================================================
# SKILL 3: Directory Bruteforce
# =============================================================================
SKILL3_DIR_BRUTE = [
    Goal(
        id="skill3_dvwa_dirs",
        type=GoalType.ENUMERATE,
        description="Discover hidden directories on DVWA using gobuster",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Find hidden directories and files",
        hints=["Use gobuster with common wordlist", "Target http://192.168.0.41:8001"],
        max_steps=3,
    ),
    Goal(
        id="skill3_bwapp_dirs",
        type=GoalType.ENUMERATE,
        description="Discover hidden directories on bWAPP using gobuster",
        target=Target(ip="192.168.0.41", hostname="bwapp",
                     known_services=[{"port": 8003, "service": "http"}]),
        success_criteria="Find hidden directories and files",
        hints=["Use gobuster with common wordlist", "Target http://192.168.0.41:8003"],
        max_steps=3,
    ),
    Goal(
        id="skill3_mutillidae_dirs",
        type=GoalType.ENUMERATE,
        description="Discover hidden directories on Mutillidae using gobuster",
        target=Target(ip="192.168.0.41", hostname="mutillidae",
                     known_services=[{"port": 8006, "service": "http"}]),
        success_criteria="Find hidden directories and files",
        hints=["Use gobuster with common wordlist", "Target http://192.168.0.41:8006"],
        max_steps=3,
    ),
    Goal(
        id="skill3_webgoat_dirs",
        type=GoalType.ENUMERATE,
        description="Discover hidden directories on WebGoat using gobuster",
        target=Target(ip="192.168.0.41", hostname="webgoat",
                     known_services=[{"port": 8004, "service": "http"}]),
        success_criteria="Find hidden directories and files",
        hints=["Use gobuster with common wordlist", "Target http://192.168.0.41:8004"],
        max_steps=3,
    ),
]

# =============================================================================
# SKILL 4: Default Credential Testing
# =============================================================================
SKILL4_DEFAULT_CREDS = [
    Goal(
        id="skill4_msf_ssh_creds",
        type=GoalType.CREDENTIAL,
        description="Test default SSH credentials on Metasploitable2",
        target=Target(ip="192.168.0.41", hostname="metasploitable",
                     known_services=[{"port": 8022, "service": "ssh"}]),
        success_criteria="Successfully authenticate with default credentials",
        hints=[
            "SSH on port 8022",
            "Try msfadmin:msfadmin",
            "Use curl or nc to test, or searchsploit for info",
        ],
        max_steps=4,
    ),
    Goal(
        id="skill4_msf_ftp_creds",
        type=GoalType.CREDENTIAL,
        description="Test FTP access on Metasploitable2",
        target=Target(ip="192.168.0.41", hostname="metasploitable",
                     known_services=[{"port": 8021, "service": "ftp"}]),
        success_criteria="Test anonymous or default FTP access",
        hints=[
            "FTP on port 8021",
            "Try anonymous login",
            "Use curl ftp://",
        ],
        max_steps=4,
    ),
    Goal(
        id="skill4_dvwa_login",
        type=GoalType.CREDENTIAL,
        description="Find default credentials for DVWA login",
        target=Target(ip="192.168.0.41", hostname="dvwa",
                     known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Identify or test default DVWA credentials",
        hints=[
            "DVWA default: admin/password",
            "Check login page at /login.php",
            "Use curl to test",
        ],
        max_steps=4,
    ),
]

# =============================================================================
# SKILL 5: API Enumeration
# =============================================================================
SKILL5_API_ENUM = [
    Goal(
        id="skill5_vampi_api",
        type=GoalType.ENUMERATE,
        description="Enumerate VAmPI REST API endpoints",
        target=Target(ip="192.168.0.41", hostname="vampi",
                     known_services=[{"port": 8008, "service": "http"}]),
        success_criteria="Discover API endpoints and documentation",
        hints=[
            "Use curl to probe endpoints",
            "Check /openapi.json or /swagger",
            "Try common API paths: /api, /v1, /users",
        ],
        max_steps=4,
    ),
    Goal(
        id="skill5_juice_api",
        type=GoalType.ENUMERATE,
        description="Enumerate Juice Shop REST API endpoints",
        target=Target(ip="192.168.0.41", hostname="juice-shop",
                     known_services=[{"port": 8002, "service": "http"}]),
        success_criteria="Discover API endpoints",
        hints=[
            "Use curl to probe endpoints",
            "Check /api, /rest",
            "Look for /api/Products, /api/Users",
        ],
        max_steps=4,
    ),
    Goal(
        id="skill5_webgoat_api",
        type=GoalType.ENUMERATE,
        description="Enumerate WebGoat API structure",
        target=Target(ip="192.168.0.41", hostname="webgoat",
                     known_services=[{"port": 8004, "service": "http"}]),
        success_criteria="Discover API or application endpoints",
        hints=[
            "Use curl to probe",
            "Check /WebGoat path",
            "Look for REST endpoints",
        ],
        max_steps=4,
    ),
]

# Combine all skills
ALL_SKILLS = {
    "skill1_port_enum": SKILL1_PORT_ENUM,
    "skill2_web_scan": SKILL2_WEB_SCAN,
    "skill3_dir_brute": SKILL3_DIR_BRUTE,
    "skill4_default_creds": SKILL4_DEFAULT_CREDS,
    "skill5_api_enum": SKILL5_API_ENUM,
}


async def collect_multiskill_chains(
    runs_per_goal: int = 3,
    model: str = "openai:gpt-4o",
    output_dir: str = None,
):
    """Collect chains across multiple skills."""
    print("=" * 70)
    print("MULTI-SKILL CHAIN COLLECTION")
    print("=" * 70)

    total_goals = sum(len(goals) for goals in ALL_SKILLS.values())
    print(f"Skills: {len(ALL_SKILLS)}")
    print(f"Goals per skill: {[len(g) for g in ALL_SKILLS.values()]}")
    print(f"Total goals: {total_goals}")
    print(f"Runs per goal: {runs_per_goal}")
    print(f"Expected chains: {total_goals * runs_per_goal}")
    print(f"Model: {model}")
    print("=" * 70)

    if output_dir is None:
        output_dir = Path(__file__).parent / "multiskill_chains"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    agent = SecurityAgent(model_name=model)
    verifier = GoalVerifier()
    similarity_scorer = ChainSimilarityScorer()

    all_chains = []
    skill_chains = {skill: [] for skill in ALL_SKILLS}
    skill_stats = {skill: {"total": 0, "successful": 0} for skill in ALL_SKILLS}

    for skill_name, goals in ALL_SKILLS.items():
        print(f"\n{'=' * 70}")
        print(f"SKILL: {skill_name}")
        print(f"{'=' * 70}")

        for goal in goals:
            print(f"\n--- {goal.id} ---")
            print(f"Goal: {goal.description}")

            for run_idx in range(runs_per_goal):
                print(f"  Run {run_idx + 1}/{runs_per_goal}...", end=" ", flush=True)
                skill_stats[skill_name]["total"] += 1

                try:
                    memory = await agent.run(goal, verbose=False)
                    verification = verifier.verify(goal, memory.tool_results)
                    chain = extract_training_chain(memory, verification)

                    # Add skill label
                    chain_data = {
                        "skill": skill_name,
                        "goal_id": chain.goal_id,
                        "goal_description": chain.goal_description,
                        "target_ip": chain.target_ip,
                        "actions": chain.actions,
                        "success": chain.success,
                        "duration_seconds": chain.duration_seconds,
                    }

                    all_chains.append(chain_data)
                    skill_chains[skill_name].append(chain_data)

                    if chain.success:
                        skill_stats[skill_name]["successful"] += 1

                    status = "✓" if chain.success else "✗"
                    tools = [a["tool"] for a in chain.actions]
                    print(f"{status} {len(chain.actions)} steps: {' → '.join(tools[:3])}")

                except Exception as e:
                    print(f"✗ Error: {str(e)[:40]}")

        # Skill summary
        stats = skill_stats[skill_name]
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        print(f"\n{skill_name} Result: {stats['successful']}/{stats['total']} ({rate:.0%})")

    # Overall summary
    print(f"\n{'=' * 70}")
    print("COLLECTION SUMMARY")
    print("=" * 70)

    total = len(all_chains)
    successful = sum(1 for c in all_chains if c["success"])

    print(f"\nTotal chains: {total}")
    print(f"Successful: {successful} ({100*successful/total:.0f}%)")

    print("\nBy skill:")
    for skill_name, stats in skill_stats.items():
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {skill_name}: {stats['successful']}/{stats['total']} ({rate:.0%})")

    # Save all chains
    all_path = output_dir / "all_chains.json"
    with open(all_path, "w") as f:
        json.dump(all_chains, f, indent=2)

    # Save by skill
    for skill_name, chains in skill_chains.items():
        skill_path = output_dir / f"{skill_name}_chains.json"
        with open(skill_path, "w") as f:
            json.dump(chains, f, indent=2)

    # Save successful only
    successful_chains = [c for c in all_chains if c["success"]]
    success_path = output_dir / "successful_chains.json"
    with open(success_path, "w") as f:
        json.dump(successful_chains, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  - all_chains.json ({len(all_chains)} chains)")
    print(f"  - successful_chains.json ({len(successful_chains)} chains)")
    for skill_name in ALL_SKILLS:
        print(f"  - {skill_name}_chains.json")

    return all_chains


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", "-r", type=int, default=3)
    parser.add_argument("--model", "-m", default="openai:gpt-4o")
    args = parser.parse_args()

    asyncio.run(collect_multiskill_chains(
        runs_per_goal=args.runs,
        model=args.model,
    ))
