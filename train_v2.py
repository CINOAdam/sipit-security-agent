#!/usr/bin/env python3
"""
SipIt V2 Training for Security Agent.

Fine-tunes a base model on reproducible, successful chains.
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
CHAINS_PATH = Path(__file__).parent / "v2_chains" / "reproducible_chains.json"
OUTPUT_DIR = Path(__file__).parent / "v2_model"


def load_chains() -> List[Dict[str, Any]]:
    """Load reproducible chains."""
    with open(CHAINS_PATH) as f:
        return json.load(f)


def chain_to_conversation(chain: Dict[str, Any]) -> str:
    """Convert a chain to a training conversation."""
    goal = chain["goal_description"]
    target = chain["target_ip"]

    # Build conversation
    messages = []

    # System message
    system = f"""You are an expert penetration tester. Your goal is to: {goal}
Target: {target}

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON."""

    # User starts with goal
    user_msg = f"Begin enumeration of the target. What's your first action?"

    # Build assistant responses from actions
    conversation = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"

    for i, action in enumerate(chain["actions"]):
        tool = action["tool"]
        params = action["parameters"]

        # Assistant response
        response = json.dumps({
            "reasoning": f"Using {tool} for reconnaissance",
            "tool_name": tool,
            "parameters": params,
            "is_complete": False
        }, indent=2)

        conversation += f" {response}</s>"

        # Next turn (if not last action)
        if i < len(chain["actions"]) - 1:
            result_status = "succeeded" if action.get("success", True) else "completed with issues"
            next_prompt = f"<s>[INST] Tool {tool} {result_status}. What's next? [/INST]"
            conversation += next_prompt

    # Final completion message
    conversation += f"<s>[INST] Enumeration complete. Summarize findings. [/INST] "
    conversation += json.dumps({
        "reasoning": "Enumeration complete - identified services and potential vulnerabilities",
        "tool_name": "none",
        "parameters": {},
        "is_complete": True
    }) + "</s>"

    return conversation


def prepare_dataset(chains: List[Dict[str, Any]]) -> Dataset:
    """Prepare dataset for training."""
    conversations = []

    for chain in chains:
        conv = chain_to_conversation(chain)
        conversations.append({"text": conv})

    return Dataset.from_list(conversations)


def train(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """Run SipIt V2 training."""
    print("=" * 70)
    print("SipIt V2 TRAINING - Security Agent")
    print("=" * 70)

    # Load chains
    print(f"\nLoading chains from {CHAINS_PATH}")
    chains = load_chains()
    print(f"Loaded {len(chains)} reproducible chains")

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

    # LoRA config
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
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
        "chains_used": len(chains),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 70)
    print("SipIt V2 TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {final_path}")

    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train(
        base_model=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
