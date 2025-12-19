#!/usr/bin/env python3
"""
Compare V2-trained model against baseline.

Tests both models on generating tool sequences for pentesting goals.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
V2_MODEL_PATH = Path(__file__).parent / "v2_model" / "final"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def load_models():
    """Load both base and V2-trained models."""
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading V2-trained model...")
    v2_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    v2_model = PeftModel.from_pretrained(v2_model, str(V2_MODEL_PATH))

    return tokenizer, base_model, v2_model


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    """Generate a response from a model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def create_test_prompts() -> List[Dict[str, str]]:
    """Create test prompts for comparison."""
    return [
        {
            "id": "juice_enum",
            "prompt": """<s>[INST] <<SYS>>
You are an expert penetration tester. Your goal is to: Enumerate Juice Shop to find API endpoints and vulnerabilities
Target: 192.168.0.41

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON.
<</SYS>>

Begin enumeration of the target. What's your first action? [/INST]""",
        },
        {
            "id": "dvwa_enum",
            "prompt": """<s>[INST] <<SYS>>
You are an expert penetration tester. Your goal is to: Enumerate DVWA to identify all attack surfaces
Target: 192.168.0.41

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON.
<</SYS>>

Begin enumeration of the target. What's your first action? [/INST]""",
        },
        {
            "id": "vampi_enum",
            "prompt": """<s>[INST] <<SYS>>
You are an expert penetration tester. Your goal is to: Enumerate VAmPI REST API to find endpoints
Target: 192.168.0.41

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON.
<</SYS>>

Begin enumeration of the target. What's your first action? [/INST]""",
        },
    ]


def extract_tool_from_response(response: str) -> str:
    """Try to extract the tool name from a response."""
    response_lower = response.lower()
    tools = ["nmap", "nikto", "gobuster", "curl", "nc", "searchsploit"]

    for tool in tools:
        if tool in response_lower:
            return tool

    # Try to parse JSON
    try:
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            data = json.loads(json_match.group())
            if "tool_name" in data:
                return data["tool_name"]
    except:
        pass

    return "unknown"


def run_comparison():
    """Run comparison between base and V2 models."""
    print("=" * 70)
    print("SipIt V2 COMPARISON")
    print("=" * 70)

    tokenizer, base_model, v2_model = load_models()
    prompts = create_test_prompts()

    results = []

    for prompt_data in prompts:
        print(f"\n{'─' * 70}")
        print(f"Test: {prompt_data['id']}")
        print(f"{'─' * 70}")

        prompt = prompt_data["prompt"]

        # Base model
        print("\n[BASE MODEL]")
        base_response = generate_response(base_model, tokenizer, prompt)
        base_tool = extract_tool_from_response(base_response)
        print(f"Response: {base_response[:200]}...")
        print(f"Tool selected: {base_tool}")

        # V2 model
        print("\n[V2-TRAINED MODEL]")
        v2_response = generate_response(v2_model, tokenizer, prompt)
        v2_tool = extract_tool_from_response(v2_response)
        print(f"Response: {v2_response[:200]}...")
        print(f"Tool selected: {v2_tool}")

        # Compare
        # Expected from training data: nmap or nikto first
        expected_tools = ["nmap", "nikto"]
        base_correct = base_tool in expected_tools
        v2_correct = v2_tool in expected_tools

        results.append({
            "test_id": prompt_data["id"],
            "base_tool": base_tool,
            "v2_tool": v2_tool,
            "base_correct": base_correct,
            "v2_correct": v2_correct,
        })

        print(f"\nExpected first tool: nmap or nikto")
        print(f"Base correct: {'✓' if base_correct else '✗'}")
        print(f"V2 correct: {'✓' if v2_correct else '✗'}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    base_score = sum(1 for r in results if r["base_correct"]) / len(results)
    v2_score = sum(1 for r in results if r["v2_correct"]) / len(results)

    print(f"\nBase model accuracy: {base_score:.1%}")
    print(f"V2-trained accuracy: {v2_score:.1%}")
    print(f"Improvement: {(v2_score - base_score):+.1%}")

    # Tool distribution
    print("\nTool selections:")
    print(f"  Base: {[r['base_tool'] for r in results]}")
    print(f"  V2:   {[r['v2_tool'] for r in results]}")

    return results


if __name__ == "__main__":
    run_comparison()
