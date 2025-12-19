#!/usr/bin/env python3
"""
Multi-skill SipIt training for interpretability research.

Trains on all chains (not just successful) to learn skill-specific patterns.
The goal is to study how different skills form distinct representations.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Paths
CHAINS_PATH = Path(__file__).parent / "multiskill_chains" / "all_chains.json"
OUTPUT_DIR = Path(__file__).parent / "multiskill_model"


def load_chains() -> List[Dict[str, Any]]:
    """Load all chains (including failures for pattern learning)."""
    with open(CHAINS_PATH) as f:
        return json.load(f)


def chain_to_conversation(chain: Dict[str, Any]) -> str:
    """Convert a chain to a training conversation with skill label."""
    skill = chain["skill"]
    goal = chain["goal_description"]
    target = chain["target_ip"]

    # System message includes skill type
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
            next_prompt = f"<s>[INST] Tool {tool} executed. Continue with {skill}. [/INST]"
            conversation += next_prompt

    # Final completion
    conversation += f"<s>[INST] {skill} task complete. Summarize. [/INST] "
    conversation += json.dumps({
        "skill": skill,
        "reasoning": f"{skill} complete",
        "tool_name": "none",
        "parameters": {},
        "is_complete": True
    }) + "</s>"

    return conversation


def prepare_dataset(chains: List[Dict[str, Any]]) -> Dataset:
    """Prepare dataset with skill labels."""
    conversations = []

    for chain in chains:
        conv = chain_to_conversation(chain)
        conversations.append({
            "text": conv,
            "skill": chain["skill"],
        })

    return Dataset.from_list(conversations)


def train(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """Run multi-skill training."""
    print("=" * 70)
    print("MULTI-SKILL SipIt TRAINING")
    print("=" * 70)

    # Load chains
    print(f"\nLoading chains from {CHAINS_PATH}")
    chains = load_chains()
    print(f"Loaded {len(chains)} chains")

    # Count by skill
    skill_counts = {}
    for chain in chains:
        skill = chain["skill"]
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

    print("\nChains by skill:")
    for skill, count in skill_counts.items():
        print(f"  {skill}: {count}")

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(chains)
    print(f"Dataset size: {len(dataset)}")

    # Load model
    print(f"\nLoading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config - slightly larger for multi-skill
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # Larger rank for more skills
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        max_length=max_seq_length,
        dataset_text_field="text",
    )

    # Trainer
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()
    print(f"\nTraining completed in {duration:.1f}s")

    # Save
    final_path = OUTPUT_DIR / "final"
    print(f"\nSaving model to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save training info
    info = {
        "base_model": base_model,
        "total_chains": len(chains),
        "chains_by_skill": skill_counts,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_r": 32,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 70)
    print("MULTI-SKILL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {final_path}")
    print(f"Skills learned: {list(skill_counts.keys())}")

    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train(
        base_model=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
