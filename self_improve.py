#!/usr/bin/env python3
"""
Self-Improvement Loop for SipIt.

The model improves by:
1. Attempting goals with current model
2. Collecting successful chains
3. Retraining on accumulated successes
4. Repeating

This is the core self-improvement hypothesis: models can get better
by learning from their own successful experiences.
"""

import json
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from .local_agent import LocalSecurityAgent, LocalAgentMemory
from .goals import Goal, GoalType, Target
from .verification import GoalVerifier, VerificationResult

# Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ITERATIONS_DIR = Path(__file__).parent / "self_improve_iterations"
INITIAL_ADAPTER = Path(__file__).parent / "multiskill_model" / "final"


# Goals for self-improvement (mix of difficulties)
SELF_IMPROVE_GOALS = [
    # Port enumeration (model should be good at this)
    Goal(
        id="si_ports_1",
        type=GoalType.ENUMERATE,
        description="Enumerate open ports and services on DVWA",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Identify open ports and versions",
        hints=["Use nmap with version detection"],
        max_steps=3,
    ),
    Goal(
        id="si_ports_2",
        type=GoalType.ENUMERATE,
        description="Scan Juice Shop for open ports",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8002, "service": "http"}]),
        success_criteria="Identify open ports",
        hints=["Use nmap"],
        max_steps=3,
    ),

    # Web scanning
    Goal(
        id="si_webscan_1",
        type=GoalType.ENUMERATE,
        description="Scan DVWA for web vulnerabilities using nikto",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Find vulnerabilities",
        hints=["Use nikto"],
        max_steps=3,
    ),
    Goal(
        id="si_webscan_2",
        type=GoalType.ENUMERATE,
        description="Perform web vulnerability scan on bWAPP",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8003, "service": "http"}]),
        success_criteria="Identify web issues",
        hints=["Use nikto"],
        max_steps=3,
    ),

    # Directory bruteforce
    Goal(
        id="si_dirs_1",
        type=GoalType.ENUMERATE,
        description="Discover hidden directories on DVWA using gobuster",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8001, "service": "http"}]),
        success_criteria="Find hidden paths",
        hints=["Use gobuster"],
        max_steps=3,
    ),

    # Credential testing
    Goal(
        id="si_creds_1",
        type=GoalType.CREDENTIAL,
        description="Test default FTP credentials on Metasploitable",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8021, "service": "ftp"}]),
        success_criteria="Test FTP access",
        hints=["Use nc or curl"],
        max_steps=4,
    ),

    # API enumeration
    Goal(
        id="si_api_1",
        type=GoalType.ENUMERATE,
        description="Enumerate REST API endpoints on VAmPI",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8008, "service": "http"}]),
        success_criteria="Find API endpoints",
        hints=["Use curl to probe /api"],
        max_steps=4,
    ),
    Goal(
        id="si_api_2",
        type=GoalType.ENUMERATE,
        description="Discover API documentation on Juice Shop",
        target=Target(ip="192.168.0.41", known_services=[{"port": 8002, "service": "http"}]),
        success_criteria="Find API or swagger docs",
        hints=["Use curl"],
        max_steps=4,
    ),
]


def chain_to_training_text(
    chain: Dict[str, Any],
    skill: str,
) -> str:
    """Convert a chain to training format."""
    goal = chain["goal_description"]
    target = chain["target_ip"]

    system = f"""You are an expert penetration tester.
Skill type: {skill}
Goal: {goal}
Target: {target}

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON."""

    user_msg = f"Begin the {skill.replace('_', ' ')} task. What's your first action?"
    conversation = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"

    for i, action in enumerate(chain["actions"]):
        tool = action["tool"]
        params = action["parameters"]

        response = json.dumps({
            "skill": skill,
            "reasoning": f"Using {tool} for {skill.replace('_', ' ')}",
            "tool_name": tool,
            "parameters": params,
            "is_complete": False
        }, indent=2)

        conversation += f" {response}</s>"

        if i < len(chain["actions"]) - 1:
            conversation += f"<s>[INST] Tool executed. Continue. [/INST]"

    # Completion marker
    conversation += f"<s>[INST] Task complete. [/INST] "
    conversation += json.dumps({
        "skill": skill,
        "reasoning": "Task complete",
        "tool_name": "none",
        "parameters": {},
        "is_complete": True
    }) + "</s>"

    return conversation


def infer_skill(goal: Goal) -> str:
    """Infer skill from goal."""
    desc = goal.description.lower()
    if "port" in desc:
        return "skill1_port_enum"
    elif "vulnerab" in desc or "nikto" in desc or "web" in desc:
        return "skill2_web_scan"
    elif "director" in desc or "gobuster" in desc:
        return "skill3_dir_brute"
    elif "credential" in desc or "ftp" in desc or "ssh" in desc:
        return "skill4_default_creds"
    elif "api" in desc:
        return "skill5_api_enum"
    return "skill1_port_enum"


async def collect_with_model(
    adapter_path: Path,
    goals: List[Goal],
    runs_per_goal: int = 2,
    is_initial: bool = False,
) -> List[Dict[str, Any]]:
    """Collect chains using the current model."""
    print(f"\nCollecting chains with adapter: {adapter_path}")

    # Use 8bit only for initial adapter (trained with 8bit compatible)
    # New adapters use float16 for compatibility
    use_8bit = is_initial

    # Create agent once and reuse
    agent = LocalSecurityAgent(adapter_path=adapter_path, use_8bit=use_8bit)
    agent.load_model()  # Load once
    verifier = GoalVerifier()

    all_chains = []
    successful = 0

    for goal in goals:
        print(f"\n  Goal: {goal.id}")

        for run in range(runs_per_goal):
            print(f"    Run {run+1}/{runs_per_goal}...", end=" ", flush=True)

            try:
                memory = await agent.run(goal, verbose=False)
                verification = verifier.verify(goal, memory.tool_results)

                chain = {
                    "goal_id": goal.id,
                    "goal_description": goal.description,
                    "target_ip": goal.target.ip,
                    "skill": infer_skill(goal),
                    "actions": memory.to_chain(),
                    "success": verification.result == VerificationResult.SUCCESS,
                }

                all_chains.append(chain)

                if chain["success"]:
                    successful += 1
                    tools = [a["tool"] for a in chain["actions"]]
                    print(f"✓ {' → '.join(tools)}")
                else:
                    tools = [a["tool"] for a in chain["actions"]]
                    print(f"✗ {' → '.join(tools)}")

            except Exception as e:
                print(f"✗ Error: {str(e)[:30]}")

    # Clean up to free memory
    del agent.model
    agent.model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    rate = 100*successful/len(all_chains) if all_chains else 0
    print(f"\n  Collected: {len(all_chains)} chains, {successful} successful ({rate:.0f}%)")

    return all_chains


def train_on_chains(
    chains: List[Dict[str, Any]],
    output_dir: Path,
    base_adapter: Path = None,
    epochs: int = 3,
) -> Path:
    """Train model on successful chains."""
    print(f"\nTraining on {len(chains)} chains...")

    # Prepare dataset
    texts = []
    for chain in chains:
        text = chain_to_training_text(chain, chain["skill"])
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)
    print(f"  Dataset size: {len(dataset)}")

    # Load model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # If we have a base adapter, merge it first
    if base_adapter and base_adapter.exists():
        print(f"  Loading base adapter from {base_adapter}...")
        model = PeftModel.from_pretrained(model, str(base_adapter))
        model = model.merge_and_unload()

    # Apply fresh LoRA
    print("  Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Training
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no",
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        max_length=2048,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("  Training...")
    trainer.train()

    # Save
    final_path = output_dir / "adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"  Saved to: {final_path}")
    return final_path


async def run_self_improvement(
    iterations: int = 3,
    runs_per_goal: int = 2,
    min_chains_to_train: int = 2,
):
    """Run the self-improvement loop."""
    print("=" * 70)
    print("SELF-IMPROVEMENT LOOP")
    print("=" * 70)
    print(f"Iterations: {iterations}")
    print(f"Goals: {len(SELF_IMPROVE_GOALS)}")
    print(f"Runs per goal: {runs_per_goal}")
    print("=" * 70)

    ITERATIONS_DIR.mkdir(exist_ok=True)

    # Start with initial adapter
    current_adapter = INITIAL_ADAPTER

    iteration_stats = []

    for iteration in range(iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{iterations}")
        print(f"{'='*70}")

        iter_dir = ITERATIONS_DIR / f"iter_{iteration + 1}"
        iter_dir.mkdir(exist_ok=True)

        # Collect chains with current model
        print("\n1. COLLECTING CHAINS")
        chains = await collect_with_model(
            adapter_path=current_adapter,
            goals=SELF_IMPROVE_GOALS,
            runs_per_goal=runs_per_goal,
            is_initial=(iteration == 0),  # First iteration uses initial adapter
        )

        # Filter successful
        successful = [c for c in chains if c["success"]]
        success_rate = len(successful) / len(chains) if chains else 0

        print(f"\n   Success rate: {len(successful)}/{len(chains)} ({success_rate:.0%})")

        # Save chains
        chains_path = iter_dir / "chains.json"
        with open(chains_path, "w") as f:
            json.dump(chains, f, indent=2)

        stats = {
            "iteration": iteration + 1,
            "total_chains": len(chains),
            "successful_chains": len(successful),
            "success_rate": success_rate,
            "adapter_used": str(current_adapter),
        }

        # Train if we have enough successful chains
        if len(successful) >= min_chains_to_train:
            print("\n2. RETRAINING ON SUCCESSES")

            # Accumulate all successful chains from previous iterations
            all_successful = list(successful)  # Start with current

            for prev_iter in range(iteration):
                prev_chains_path = ITERATIONS_DIR / f"iter_{prev_iter + 1}" / "chains.json"
                if prev_chains_path.exists():
                    with open(prev_chains_path) as f:
                        prev_chains = json.load(f)
                    prev_successful = [c for c in prev_chains if c["success"]]
                    all_successful.extend(prev_successful)

            print(f"   Total successful chains for training: {len(all_successful)}")

            new_adapter = train_on_chains(
                chains=all_successful,
                output_dir=iter_dir / "model",
                base_adapter=None,  # Train fresh each time on all accumulated data
                epochs=3,
            )

            current_adapter = new_adapter
            stats["trained"] = True
            stats["training_chains"] = len(all_successful)
        else:
            print(f"\n2. SKIPPING TRAINING (need {min_chains_to_train}+ successful chains)")
            stats["trained"] = False

        iteration_stats.append(stats)

        # Save iteration summary
        summary_path = iter_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("SELF-IMPROVEMENT COMPLETE")
    print("=" * 70)

    print("\nIteration Summary:")
    print("-" * 50)
    for stats in iteration_stats:
        trained = "✓" if stats.get("trained") else "✗"
        print(f"  Iter {stats['iteration']}: {stats['success_rate']:.0%} success ({stats['successful_chains']}/{stats['total_chains']}) [trained: {trained}]")

    # Check for improvement
    if len(iteration_stats) >= 2:
        first_rate = iteration_stats[0]["success_rate"]
        last_rate = iteration_stats[-1]["success_rate"]
        delta = last_rate - first_rate

        print(f"\nImprovement: {first_rate:.0%} → {last_rate:.0%} ({delta:+.0%})")

        if delta > 0:
            print("*** MODEL IMPROVED THROUGH SELF-TRAINING! ***")
        elif delta == 0:
            print("Model performance unchanged")
        else:
            print("Model performance decreased (may need more data)")

    # Save overall summary
    summary_path = ITERATIONS_DIR / "overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "iterations": iteration_stats,
            "final_adapter": str(current_adapter),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to: {ITERATIONS_DIR}")

    return iteration_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", "-i", type=int, default=3)
    parser.add_argument("--runs", "-r", type=int, default=2)
    args = parser.parse_args()

    asyncio.run(run_self_improvement(
        iterations=args.iterations,
        runs_per_goal=args.runs,
    ))
