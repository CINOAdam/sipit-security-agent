#!/usr/bin/env python3
"""
Chain-based skill pattern analysis for interpretability research.

Analyzes behavioral patterns from collected chains:
1. Tool usage patterns per skill
2. Action sequence similarity
3. Skill distinctiveness metrics
4. Failure mode analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib.pyplot as plt

CHAINS_PATH = Path(__file__).parent / "multiskill_chains" / "all_chains.json"
OUTPUT_DIR = Path(__file__).parent / "interpretability_results"


def load_chains() -> List[Dict[str, Any]]:
    """Load all collected chains."""
    with open(CHAINS_PATH) as f:
        return json.load(f)


def analyze_tool_patterns(chains: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Analyze tool usage frequency per skill."""
    skill_tools = defaultdict(Counter)

    for chain in chains:
        skill = chain["skill"]
        for action in chain["actions"]:
            skill_tools[skill][action["tool"]] += 1

    # Normalize to frequencies
    tool_frequencies = {}
    for skill, counts in skill_tools.items():
        total = sum(counts.values())
        tool_frequencies[skill] = {tool: count/total for tool, count in counts.items()}

    return tool_frequencies


def compute_skill_signatures(tool_frequencies: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Create vector signatures for each skill based on tool usage."""
    all_tools = sorted(set(
        tool for freqs in tool_frequencies.values()
        for tool in freqs.keys()
    ))

    signatures = {}
    for skill, freqs in tool_frequencies.items():
        vec = np.array([freqs.get(tool, 0) for tool in all_tools])
        signatures[skill] = vec

    return signatures, all_tools


def compute_skill_similarity(signatures: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute cosine similarity between skill signatures."""
    skills = sorted(signatures.keys())
    n = len(skills)
    similarity = np.zeros((n, n))

    for i, skill_i in enumerate(skills):
        for j, skill_j in enumerate(skills):
            vec_i = signatures[skill_i]
            vec_j = signatures[skill_j]

            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)

            if norm_i > 0 and norm_j > 0:
                similarity[i, j] = np.dot(vec_i, vec_j) / (norm_i * norm_j)
            else:
                similarity[i, j] = 0

    return similarity, skills


def analyze_first_tool(chains: List[Dict[str, Any]]) -> Dict[str, Counter]:
    """Analyze which tool is used first for each skill."""
    first_tools = defaultdict(Counter)

    for chain in chains:
        skill = chain["skill"]
        if chain["actions"]:
            first_tool = chain["actions"][0]["tool"]
            first_tools[skill][first_tool] += 1

    return first_tools


def analyze_action_sequences(chains: List[Dict[str, Any]]) -> Dict[str, Counter]:
    """Analyze common tool sequences per skill."""
    sequences = defaultdict(Counter)

    for chain in chains:
        skill = chain["skill"]
        tools = [a["tool"] for a in chain["actions"]]
        seq = " → ".join(tools[:3])  # First 3 tools
        sequences[skill][seq] += 1

    return sequences


def analyze_failure_modes(chains: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze failure patterns per skill."""
    skill_stats = defaultdict(lambda: {"total": 0, "success": 0, "failure_patterns": Counter()})

    for chain in chains:
        skill = chain["skill"]
        skill_stats[skill]["total"] += 1

        if chain["success"]:
            skill_stats[skill]["success"] += 1
        else:
            # Record failure pattern (tool sequence)
            tools = [a["tool"] for a in chain["actions"]]
            pattern = " → ".join(tools)
            skill_stats[skill]["failure_patterns"][pattern] += 1

    return dict(skill_stats)


def visualize_tool_distribution(tool_frequencies: Dict[str, Dict[str, float]], output_path: Path):
    """Create heatmap of tool usage across skills."""
    all_tools = sorted(set(
        tool for freqs in tool_frequencies.values()
        for tool in freqs.keys()
    ))

    skills = sorted(tool_frequencies.keys())
    skill_labels = [s.replace("_", "\n") for s in skills]

    # Create matrix
    matrix = np.zeros((len(skills), len(all_tools)))
    for i, skill in enumerate(skills):
        for j, tool in enumerate(all_tools):
            matrix[i, j] = tool_frequencies[skill].get(tool, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(all_tools)))
    ax.set_xticklabels(all_tools, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels(skill_labels, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Tool Usage Frequency", fontsize=11)

    # Add value annotations
    for i in range(len(skills)):
        for j in range(len(all_tools)):
            val = matrix[i, j]
            if val > 0.05:
                text = f"{val:.0%}"
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

    ax.set_title("Tool Usage by Skill Type", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def visualize_skill_similarity(similarity: np.ndarray, skills: List[str], output_path: Path):
    """Create skill similarity matrix visualization."""
    skill_labels = [s.replace("_", "\n") for s in skills]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(similarity, cmap='RdYlBu', vmin=0, vmax=1)

    ax.set_xticks(range(len(skills)))
    ax.set_xticklabels(skill_labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels(skill_labels, fontsize=9)

    # Add value annotations
    for i in range(len(skills)):
        for j in range(len(skills)):
            val = similarity[i, j]
            color = "white" if val > 0.7 or val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=11)

    ax.set_title("Skill Similarity Matrix\n(based on tool usage patterns)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def visualize_first_tool(first_tools: Dict[str, Counter], output_path: Path):
    """Visualize first tool selection by skill."""
    skills = sorted(first_tools.keys())
    all_tools = sorted(set(
        tool for counts in first_tools.values()
        for tool in counts.keys()
    ))

    skill_labels = [s.replace("skill", "S").replace("_", " ") for s in skills]

    fig, axes = plt.subplots(1, len(skills), figsize=(15, 4), sharey=True)

    colors = {
        'nmap': '#e41a1c',
        'nikto': '#377eb8',
        'gobuster': '#4daf4a',
        'nc': '#984ea3',
        'curl': '#ff7f00',
        'searchsploit': '#ffff33',
    }

    for idx, (ax, skill) in enumerate(zip(axes, skills)):
        counts = first_tools[skill]
        total = sum(counts.values())

        tools = []
        values = []
        cols = []

        for tool in all_tools:
            if counts.get(tool, 0) > 0:
                tools.append(tool)
                values.append(counts[tool] / total)
                cols.append(colors.get(tool, '#999999'))

        bars = ax.bar(tools, values, color=cols)
        ax.set_title(skill_labels[idx], fontsize=10)
        ax.set_ylim(0, 1)

        if idx == 0:
            ax.set_ylabel("Frequency", fontsize=11)

        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, values):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.0%}', ha='center', fontsize=8)

    fig.suptitle("First Tool Selection by Skill", fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    """Run chain-based interpretability analysis."""
    print("=" * 70)
    print("CHAIN-BASED SKILL PATTERN ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load chains
    print("\n1. Loading chains...")
    chains = load_chains()
    print(f"   Total chains: {len(chains)}")

    skill_counts = Counter(c["skill"] for c in chains)
    for skill, count in sorted(skill_counts.items()):
        print(f"   {skill}: {count}")

    # Tool usage patterns
    print("\n2. Analyzing tool usage patterns...")
    tool_frequencies = analyze_tool_patterns(chains)

    print("\n   Tool distribution by skill:")
    for skill, freqs in sorted(tool_frequencies.items()):
        top_tools = sorted(freqs.items(), key=lambda x: -x[1])[:3]
        tools_str = ", ".join([f"{t}:{v:.0%}" for t, v in top_tools])
        print(f"   {skill}: {tools_str}")

    # Skill signatures and similarity
    print("\n3. Computing skill signatures...")
    signatures, all_tools = compute_skill_signatures(tool_frequencies)
    similarity, skills = compute_skill_similarity(signatures)

    print("\n   Skill similarity (tool-based):")
    for i, skill_i in enumerate(skills):
        for j, skill_j in enumerate(skills):
            if i < j:
                sim = similarity[i, j]
                print(f"   {skill_i} ↔ {skill_j}: {sim:.2f}")

    # First tool analysis
    print("\n4. Analyzing first tool selection...")
    first_tools = analyze_first_tool(chains)

    print("\n   First tool by skill:")
    for skill, counts in sorted(first_tools.items()):
        top = counts.most_common(1)[0]
        total = sum(counts.values())
        print(f"   {skill}: {top[0]} ({top[1]/total:.0%})")

    # Action sequences
    print("\n5. Analyzing action sequences...")
    sequences = analyze_action_sequences(chains)

    print("\n   Most common sequences by skill:")
    for skill, seqs in sorted(sequences.items()):
        top_seq = seqs.most_common(1)[0]
        print(f"   {skill}: {top_seq[0]} ({top_seq[1]}x)")

    # Failure modes
    print("\n6. Analyzing failure patterns...")
    failure_stats = analyze_failure_modes(chains)

    print("\n   Success rates and common failures:")
    for skill, stats in sorted(failure_stats.items()):
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"   {skill}: {rate:.0%} success ({stats['success']}/{stats['total']})")
        if stats["failure_patterns"]:
            top_fail = stats["failure_patterns"].most_common(1)[0]
            print(f"      Common failure: {top_fail[0]} ({top_fail[1]}x)")

    # Visualizations
    print("\n7. Creating visualizations...")
    visualize_tool_distribution(tool_frequencies, OUTPUT_DIR / "tool_distribution.png")
    visualize_skill_similarity(similarity, skills, OUTPUT_DIR / "skill_similarity.png")
    visualize_first_tool(first_tools, OUTPUT_DIR / "first_tool_selection.png")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Identify most distinctive skills
    print("\n1. SKILL DISTINCTIVENESS:")
    for i, skill in enumerate(skills):
        avg_sim = np.mean([similarity[i, j] for j in range(len(skills)) if i != j])
        print(f"   {skill}: avg similarity = {avg_sim:.2f}")
        if avg_sim < 0.3:
            print(f"      → HIGHLY DISTINCTIVE (low overlap with other skills)")
        elif avg_sim > 0.7:
            print(f"      → OVERLAPPING (similar to other skills)")

    # Identify signature tools
    print("\n2. SIGNATURE TOOLS (>50% usage):")
    for skill, freqs in sorted(tool_frequencies.items()):
        for tool, freq in freqs.items():
            if freq > 0.5:
                print(f"   {skill} → {tool} ({freq:.0%})")

    # Behavioral patterns
    print("\n3. BEHAVIORAL PATTERNS:")
    print("   - Port enumeration: Starts with nmap, systematic scanning")
    print("   - Web scanning: Nikto dominant, vulnerability-focused")
    print("   - Dir bruteforce: Gobuster-heavy, path discovery")
    print("   - Credential testing: Mixed nc/curl, trial-based")
    print("   - API enumeration: Curl-centric, endpoint probing")

    # Save results
    results = {
        "total_chains": len(chains),
        "chains_by_skill": dict(skill_counts),
        "tool_frequencies": tool_frequencies,
        "skill_similarity": similarity.tolist(),
        "skills": skills,
        "first_tool_by_skill": {s: dict(c) for s, c in first_tools.items()},
        "common_sequences": {s: list(c.most_common(3)) for s, c in sequences.items()},
        "failure_stats": {s: {"success_rate": v["success"]/v["total"]} for s, v in failure_stats.items()},
    }

    results_path = OUTPUT_DIR / "chain_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
