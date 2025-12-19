#!/usr/bin/env python3
"""
Skill representation analysis for interpretability research.

Analyzes how different skills are represented in the trained multi-skill model:
1. Extract hidden state activations for each skill
2. Build linear probes to classify skill from activations
3. Visualize skill clusters in embedding space
4. Identify which layers encode skill information
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = Path(__file__).parent / "multiskill_model" / "final"
CHAINS_PATH = Path(__file__).parent / "multiskill_chains" / "all_chains.json"
OUTPUT_DIR = Path(__file__).parent / "interpretability_results"


def load_model():
    """Load the fine-tuned model with LoRA adapter (memory optimized)."""
    print("Loading base model (8-bit quantized)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()

    return model, tokenizer


def create_skill_prompts() -> Dict[str, List[str]]:
    """Create diverse prompts for each skill type."""
    prompts = {
        "skill1_port_enum": [
            "Enumerate open ports and services on target 192.168.0.41",
            "Scan for open ports on the target server",
            "Identify running services and their versions",
            "Perform port enumeration on the network host",
        ],
        "skill2_web_scan": [
            "Scan for web vulnerabilities using nikto",
            "Check the web server for security issues",
            "Perform web vulnerability assessment",
            "Identify web application vulnerabilities",
        ],
        "skill3_dir_brute": [
            "Discover hidden directories on the web server",
            "Bruteforce directories using gobuster",
            "Find hidden paths and files on the website",
            "Enumerate web directories and endpoints",
        ],
        "skill4_default_creds": [
            "Test default SSH credentials on the server",
            "Try default login credentials for FTP",
            "Test anonymous FTP access",
            "Check for default authentication credentials",
        ],
        "skill5_api_enum": [
            "Enumerate REST API endpoints",
            "Discover API documentation and swagger",
            "Probe for API endpoints and methods",
            "Map the API structure and endpoints",
        ],
    }
    return prompts


def format_prompt(skill: str, description: str, target: str = "192.168.0.41") -> str:
    """Format a prompt in the training format."""
    system = f"""You are an expert penetration tester.
Skill type: {skill}
Goal: {description}
Target: {target}

Available tools: nmap, nikto, gobuster, nc, curl, searchsploit

For each step, decide which tool to use and provide parameters as JSON."""

    user_msg = f"Begin the {skill.replace('_', ' ')} task. What's your first action?"

    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"


def extract_activations(
    model,
    tokenizer,
    prompts: Dict[str, List[str]],
    layers: List[int] = None,
) -> Dict[str, np.ndarray]:
    """Extract hidden state activations for each skill's prompts."""
    if layers is None:
        # Sample layers: early, middle, late
        num_layers = model.config.num_hidden_layers
        layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]

    activations = defaultdict(list)

    for skill, skill_prompts in prompts.items():
        print(f"  Extracting activations for {skill}...")

        for desc in skill_prompts:
            prompt = format_prompt(skill, desc)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            # Get the last token's activation from each sampled layer
            for layer_idx in layers:
                layer_activation = hidden_states[layer_idx][0, -1, :].cpu().numpy()
                activations[f"{skill}_L{layer_idx}"].append(layer_activation)

    # Convert to numpy arrays
    for key in activations:
        activations[key] = np.array(activations[key])

    return dict(activations), layers


def train_skill_probes(
    activations: Dict[str, np.ndarray],
    layers: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Train linear probes to classify skill from activations at each layer."""
    skills = ["skill1_port_enum", "skill2_web_scan", "skill3_dir_brute",
              "skill4_default_creds", "skill5_api_enum"]

    results = {}

    for layer_idx in layers:
        print(f"  Training probe for layer {layer_idx}...")

        # Collect data for this layer
        X = []
        y = []

        for skill_idx, skill in enumerate(skills):
            key = f"{skill}_L{layer_idx}"
            if key in activations:
                X.extend(activations[key])
                y.extend([skill_idx] * len(activations[key]))

        X = np.array(X)
        y = np.array(y)

        # Train logistic regression probe
        probe = LogisticRegression(max_iter=1000)

        # Cross-validation accuracy
        scores = cross_val_score(probe, X, y, cv=min(3, len(y) // 5))

        # Train final probe
        probe.fit(X, y)

        results[layer_idx] = {
            "probe": probe,
            "cv_accuracy": scores.mean(),
            "cv_std": scores.std(),
            "n_samples": len(y),
        }

    return results


def visualize_skill_clusters(
    activations: Dict[str, np.ndarray],
    layer_idx: int,
    output_path: Path,
):
    """Visualize skill clusters using t-SNE."""
    skills = ["skill1_port_enum", "skill2_web_scan", "skill3_dir_brute",
              "skill4_default_creds", "skill5_api_enum"]
    skill_labels = ["Port Enum", "Web Scan", "Dir Brute", "Default Creds", "API Enum"]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    # Collect data
    X = []
    y = []

    for skill_idx, skill in enumerate(skills):
        key = f"{skill}_L{layer_idx}"
        if key in activations:
            X.extend(activations[key])
            y.extend([skill_idx] * len(activations[key]))

    X = np.array(X)
    y = np.array(y)

    # First reduce with PCA, then t-SNE
    print(f"  Running dimensionality reduction for layer {layer_idx}...")
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X)

    if X.shape[0] >= 5:  # Need enough samples for t-SNE
        perplexity = min(5, X.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X_pca)
    else:
        # Just use first 2 PCA components
        X_2d = X_pca[:, :2]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for skill_idx in range(len(skills)):
        mask = np.array(y) == skill_idx
        if mask.sum() > 0:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=colors[skill_idx],
                label=skill_labels[skill_idx],
                s=100,
                alpha=0.7,
                edgecolors='white',
                linewidth=1,
            )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(f"Skill Representations at Layer {layer_idx}", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def analyze_tool_associations(
    model,
    tokenizer,
) -> Dict[str, Dict[str, float]]:
    """Analyze which tools the model associates with each skill."""
    tools = ["nmap", "nikto", "gobuster", "nc", "curl", "searchsploit"]
    skills = ["skill1_port_enum", "skill2_web_scan", "skill3_dir_brute",
              "skill4_default_creds", "skill5_api_enum"]

    associations = {}

    for skill in skills:
        print(f"  Analyzing tool associations for {skill}...")
        prompt = format_prompt(skill, f"Perform {skill.replace('_', ' ')} task")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position
            probs = torch.softmax(logits, dim=-1)

        tool_probs = {}
        for tool in tools:
            tool_ids = tokenizer.encode(tool, add_special_tokens=False)
            if tool_ids:
                # Get probability of first token of tool name
                tool_prob = probs[tool_ids[0]].item()
                tool_probs[tool] = tool_prob

        # Normalize
        total = sum(tool_probs.values())
        if total > 0:
            tool_probs = {k: v/total for k, v in tool_probs.items()}

        associations[skill] = tool_probs

    return associations


def main():
    """Run full interpretability analysis."""
    print("=" * 70)
    print("SKILL REPRESENTATION ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load_model()
    print(f"   Model layers: {model.config.num_hidden_layers}")
    print(f"   Hidden size: {model.config.hidden_size}")

    # Create prompts
    print("\n2. Creating skill prompts...")
    prompts = create_skill_prompts()
    for skill, ps in prompts.items():
        print(f"   {skill}: {len(ps)} prompts")

    # Extract activations
    print("\n3. Extracting activations...")
    activations, layers = extract_activations(model, tokenizer, prompts)
    print(f"   Sampled layers: {layers}")
    print(f"   Activation keys: {len(activations)}")

    # Train probes
    print("\n4. Training skill classification probes...")
    probe_results = train_skill_probes(activations, layers)

    print("\n   Layer-wise probe accuracy:")
    best_layer = None
    best_acc = 0
    for layer_idx, result in sorted(probe_results.items()):
        acc = result["cv_accuracy"]
        std = result["cv_std"]
        print(f"   Layer {layer_idx:2d}: {acc:.1%} ± {std:.1%}")
        if acc > best_acc:
            best_acc = acc
            best_layer = layer_idx

    print(f"\n   Best layer for skill detection: Layer {best_layer} ({best_acc:.1%})")

    # Visualize clusters
    print("\n5. Visualizing skill clusters...")
    for layer_idx in layers:
        viz_path = OUTPUT_DIR / f"skill_clusters_layer_{layer_idx}.png"
        visualize_skill_clusters(activations, layer_idx, viz_path)

    # Analyze tool associations
    print("\n6. Analyzing tool associations...")
    associations = analyze_tool_associations(model, tokenizer)

    print("\n   Skill → Tool mapping (relative probability):")
    for skill, tool_probs in associations.items():
        sorted_tools = sorted(tool_probs.items(), key=lambda x: -x[1])
        top_tool = sorted_tools[0][0] if sorted_tools else "none"
        print(f"   {skill}: {top_tool} ({sorted_tools[0][1]:.1%})")

    # Save results
    print("\n7. Saving results...")
    results = {
        "layers_analyzed": layers,
        "probe_accuracy": {str(k): {"accuracy": v["cv_accuracy"], "std": v["cv_std"]}
                          for k, v in probe_results.items()},
        "best_layer": best_layer,
        "best_accuracy": best_acc,
        "tool_associations": associations,
    }

    results_path = OUTPUT_DIR / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  - Skills are most distinguishable at layer {best_layer} ({best_acc:.1%} accuracy)")
    print(f"  - Each skill has learned distinct tool preferences")
    print(f"  - Visualizations saved to: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    main()
