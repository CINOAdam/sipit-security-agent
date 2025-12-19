#!/usr/bin/env python3
"""
Generation-based skill probing - analyzes full model responses.

Tests whether the fine-tuned model generates skill-appropriate tool selections
in its JSON responses, compared to the base model.
"""

import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = Path(__file__).parent / "multiskill_model" / "final"
OUTPUT_DIR = Path(__file__).parent / "interpretability_results"

# Expected tools per skill
EXPECTED_TOOLS = {
    "skill1_port_enum": ["nmap"],
    "skill2_web_scan": ["nikto"],
    "skill3_dir_brute": ["gobuster"],
    "skill4_default_creds": ["nc", "curl"],  # Either is acceptable
    "skill5_api_enum": ["curl"],
}

SKILL_DESCRIPTIONS = {
    "skill1_port_enum": "Enumerate open ports and services",
    "skill2_web_scan": "Scan for web vulnerabilities using nikto",
    "skill3_dir_brute": "Discover hidden directories using gobuster",
    "skill4_default_creds": "Test default SSH or FTP credentials",
    "skill5_api_enum": "Enumerate REST API endpoints",
}


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


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 300) -> str:
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


def extract_tool_from_response(response: str) -> Optional[str]:
    """Extract tool_name from JSON response."""
    # Try to find JSON in response
    json_match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("tool_name")
        except json.JSONDecodeError:
            pass

    # Fallback: look for tool names in response
    tools = ["nmap", "nikto", "gobuster", "nc", "curl", "searchsploit"]
    for tool in tools:
        if tool in response.lower():
            return tool

    return None


def probe_skill(
    model,
    tokenizer,
    skill: str,
    n_runs: int = 3,
) -> List[Dict]:
    """Probe model's full generation for a skill."""
    description = SKILL_DESCRIPTIONS[skill]
    expected = EXPECTED_TOOLS[skill]

    results = []
    for i in range(n_runs):
        prompt = format_prompt(skill, description)
        response = generate_response(model, tokenizer, prompt)
        tool = extract_tool_from_response(response)

        correct = tool in expected if tool else False

        results.append({
            "run": i + 1,
            "extracted_tool": tool,
            "expected_tools": expected,
            "correct": correct,
            "response": response[:500],
        })

    return results


def main():
    """Run generation-based skill probing."""
    print("=" * 70)
    print("GENERATION-BASED SKILL PROBING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    comparison = {}

    for model_type, use_lora in [("base", False), ("finetuned", True)]:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}")
        print("=" * 70)

        model, tokenizer = load_model(use_lora=use_lora)

        all_results = {}
        stats = defaultdict(lambda: {"correct": 0, "total": 0, "tools": []})

        for skill in EXPECTED_TOOLS.keys():
            print(f"\nProbing {skill}...")
            results = probe_skill(model, tokenizer, skill)
            all_results[skill] = results

            for r in results:
                stats[skill]["total"] += 1
                stats[skill]["tools"].append(r["extracted_tool"])
                if r["correct"]:
                    stats[skill]["correct"] += 1

                status = "✓" if r["correct"] else "✗"
                print(f"  Run {r['run']}: {r['extracted_tool']} {status}")

        # Summary
        print(f"\n{'-' * 50}")
        print(f"{model_type.upper()} SUMMARY:")
        print("-" * 50)

        total_correct = 0
        total_all = 0
        skill_accuracies = {}

        for skill, s in sorted(stats.items()):
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            skill_accuracies[skill] = acc
            total_correct += s["correct"]
            total_all += s["total"]
            tools = ", ".join([t or "none" for t in s["tools"]])
            print(f"  {skill}: {acc:.0%} ({s['correct']}/{s['total']}) [{tools}]")

        overall_acc = total_correct / total_all if total_all > 0 else 0
        print(f"\n  OVERALL: {overall_acc:.0%} ({total_correct}/{total_all})")

        comparison[model_type] = {
            "overall_accuracy": overall_acc,
            "skill_accuracies": skill_accuracies,
            "results": all_results,
        }

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Comparison
    print("\n" + "=" * 70)
    print("BASE vs FINETUNED COMPARISON")
    print("=" * 70)

    print("\nSkill-wise improvement:")
    for skill in EXPECTED_TOOLS.keys():
        base_acc = comparison["base"]["skill_accuracies"].get(skill, 0)
        ft_acc = comparison["finetuned"]["skill_accuracies"].get(skill, 0)
        delta = ft_acc - base_acc

        if delta > 0:
            indicator = f"+{delta:.0%} ↑"
        elif delta < 0:
            indicator = f"{delta:.0%} ↓"
        else:
            indicator = "0% ="

        print(f"  {skill}: {base_acc:.0%} → {ft_acc:.0%} ({indicator})")

    base_overall = comparison["base"]["overall_accuracy"]
    ft_overall = comparison["finetuned"]["overall_accuracy"]
    print(f"\nOverall: {base_overall:.0%} → {ft_overall:.0%} ({ft_overall - base_overall:+.0%})")

    # Save results
    results_path = OUTPUT_DIR / "generation_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
