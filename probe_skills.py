#!/usr/bin/env python3
"""
Skill probing via generation - tests if model predicts correct tools.

This is a practical interpretability test:
1. Given a skill prompt, what tool does the model predict?
2. Does the model differentiate between skills?
3. How confident is the model in its tool selection?
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = Path(__file__).parent / "multiskill_model" / "final"
OUTPUT_DIR = Path(__file__).parent / "interpretability_results"


# Tool tokens and expected mappings
EXPECTED_TOOLS = {
    "skill1_port_enum": "nmap",
    "skill2_web_scan": "nikto",
    "skill3_dir_brute": "gobuster",
    "skill4_default_creds": "curl",  # or nc
    "skill5_api_enum": "curl",
}

ALL_TOOLS = ["nmap", "nikto", "gobuster", "nc", "curl", "searchsploit"]


def load_model(use_lora: bool = True):
    """Load model with optional LoRA adapter."""
    print(f"Loading model (LoRA={use_lora})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
    )

    if use_lora:
        print("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    model.eval()
    return model, tokenizer


def format_prompt(skill: str, description: str) -> str:
    """Format prompt in training format."""
    system = f"""You are an expert penetration tester.
Skill type: {skill}
Goal: {description}
Target: 192.168.0.41

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON."""

    user_msg = f"Begin the {skill.replace('_', ' ')} task. What's your first action?"
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"


def get_tool_logits(
    model,
    tokenizer,
    prompt: str,
) -> Dict[str, float]:
    """Get model's logit scores for each tool at the generation position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position

    # Get logits for each tool's first token
    tool_logits = {}
    for tool in ALL_TOOLS:
        tool_ids = tokenizer.encode(tool, add_special_tokens=False)
        if tool_ids:
            tool_logits[tool] = logits[tool_ids[0]].item()

    return tool_logits


def logits_to_probs(logits: Dict[str, float]) -> Dict[str, float]:
    """Convert logits to probabilities via softmax."""
    values = list(logits.values())
    max_val = max(values)
    exp_vals = {k: torch.exp(torch.tensor(v - max_val)).item() for k, v in logits.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    """Generate model response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def probe_skill(
    model,
    tokenizer,
    skill: str,
    n_prompts: int = 3,
) -> Dict:
    """Probe model's tool selection for a skill."""
    descriptions = [
        f"Perform {skill.replace('_', ' ')} on target",
        f"Execute {skill.replace('_', ' ')} assessment",
        f"Run {skill.replace('_', ' ')} task",
    ][:n_prompts]

    results = []
    for desc in descriptions:
        prompt = format_prompt(skill, desc)

        # Get logits
        logits = get_tool_logits(model, tokenizer, prompt)
        probs = logits_to_probs(logits)

        # Get top prediction
        top_tool = max(probs.items(), key=lambda x: x[1])

        # Generate actual response
        response = generate_response(model, tokenizer, prompt, max_tokens=150)

        # Check if expected tool appears in response
        expected = EXPECTED_TOOLS[skill]
        contains_expected = expected.lower() in response.lower()

        results.append({
            "description": desc,
            "top_tool": top_tool[0],
            "top_prob": top_tool[1],
            "all_probs": probs,
            "expected_tool": expected,
            "correct_prediction": top_tool[0] == expected,
            "response_contains_expected": contains_expected,
            "response_preview": response[:200],
        })

    return results


def main():
    """Run skill probing analysis."""
    print("=" * 70)
    print("SKILL PROBING ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Test both base and fine-tuned models
    for model_type, use_lora in [("base", False), ("finetuned", True)]:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}")
        print("=" * 70)

        model, tokenizer = load_model(use_lora=use_lora)

        all_results = {}
        summary_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for skill in EXPECTED_TOOLS.keys():
            print(f"\nProbing {skill}...")
            results = probe_skill(model, tokenizer, skill)
            all_results[skill] = results

            for r in results:
                summary_stats[skill]["total"] += 1
                if r["correct_prediction"]:
                    summary_stats[skill]["correct"] += 1

                print(f"  {r['description'][:40]}")
                print(f"    Top tool: {r['top_tool']} ({r['top_prob']:.1%})")
                print(f"    Expected: {r['expected_tool']}")
                print(f"    Correct: {'✓' if r['correct_prediction'] else '✗'}")

        # Summary
        print(f"\n{'-' * 50}")
        print(f"{model_type.upper()} MODEL SUMMARY:")
        print("-" * 50)

        total_correct = 0
        total_all = 0

        for skill, stats in sorted(summary_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            total_correct += stats["correct"]
            total_all += stats["total"]
            print(f"  {skill}: {acc:.0%} ({stats['correct']}/{stats['total']})")

        overall_acc = total_correct / total_all if total_all > 0 else 0
        print(f"\n  OVERALL: {overall_acc:.0%} ({total_correct}/{total_all})")

        # Save results
        results_path = OUTPUT_DIR / f"probe_results_{model_type}.json"
        with open(results_path, "w") as f:
            json.dump({
                "model_type": model_type,
                "use_lora": use_lora,
                "results": all_results,
                "summary": dict(summary_stats),
                "overall_accuracy": overall_acc,
            }, f, indent=2)
        print(f"\nSaved: {results_path}")

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "=" * 70)
    print("PROBING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
