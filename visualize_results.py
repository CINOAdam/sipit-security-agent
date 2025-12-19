#!/usr/bin/env python3
"""
Create summary visualization of interpretability findings.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "interpretability_results"


def create_comparison_chart():
    """Create bar chart comparing base vs fine-tuned model."""
    skills = [
        "Port Enum",
        "Web Scan",
        "Dir Brute",
        "Cred Test",
        "API Enum",
    ]

    base_scores = [100, 100, 100, 0, 0]  # From probe results
    finetuned_scores = [100, 100, 100, 100, 100]

    x = np.arange(len(skills))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, base_scores, width, label='Base Model',
                   color='#808080', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, finetuned_scores, width, label='Fine-tuned (SipIt)',
                   color='#2ecc71', edgecolor='black', linewidth=1)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Skill Type', fontsize=12)
    ax.set_title('Skill Differentiation: Base vs SipIt Fine-tuned Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(skills, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    # Add annotations for improvements
    ax.annotate('+100%', xy=(3, 50), fontsize=12, fontweight='bold', color='green', ha='center')
    ax.annotate('+100%', xy=(4, 50), fontsize=12, fontweight='bold', color='green', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "skill_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'skill_comparison.png'}")


def create_tool_signature_chart():
    """Create chart showing tool signatures per skill."""
    # From chain analysis
    skill_tools = {
        "Port Enum": {"nmap": 96, "nikto": 4},
        "Web Scan": {"nikto": 100},
        "Dir Brute": {"gobuster": 72, "curl": 25, "nikto": 3},
        "Cred Test": {"curl": 54, "nc": 36, "nikto": 7},
        "API Enum": {"curl": 53, "gobuster": 28, "nikto": 17},
    }

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    colors = {
        'nmap': '#e41a1c',
        'nikto': '#377eb8',
        'gobuster': '#4daf4a',
        'nc': '#984ea3',
        'curl': '#ff7f00',
    }

    for idx, (skill, tools) in enumerate(skill_tools.items()):
        ax = axes[idx]
        names = list(tools.keys())
        values = list(tools.values())
        cols = [colors.get(t, '#999999') for t in names]

        bars = ax.bar(names, values, color=cols, edgecolor='black', linewidth=1)
        ax.set_title(skill, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val}%', ha='center', fontsize=9)

        if idx == 0:
            ax.set_ylabel('Usage (%)', fontsize=11)

    fig.suptitle('Tool Signatures by Skill Type', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tool_signatures.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'tool_signatures.png'}")


def create_summary_report():
    """Create text summary of findings."""
    report = """
================================================================================
INTERPRETABILITY ANALYSIS SUMMARY
================================================================================

1. TRAINING DATA
   - 54 chains collected across 5 skill types
   - 6% overall success rate (not needed for pattern learning)
   - Each skill has distinct behavioral signature

2. SKILL SIGNATURES (Tool Usage Patterns)
   ┌────────────────────┬─────────────────────────────────────────┐
   │ Skill              │ Primary Tool(s)                         │
   ├────────────────────┼─────────────────────────────────────────┤
   │ Port Enumeration   │ nmap (96%)                              │
   │ Web Scanning       │ nikto (100%)                            │
   │ Dir Bruteforce     │ gobuster (72%), curl (25%)              │
   │ Credential Testing │ curl (54%), nc (36%)                    │
   │ API Enumeration    │ curl (53%), gobuster (28%)              │
   └────────────────────┴─────────────────────────────────────────┘

3. SKILL SIMILARITY ANALYSIS
   - Port Enum ↔ Web Scan: 0.04 (highly distinct)
   - Dir Brute ↔ API Enum: 0.71 (similar - both use curl/gobuster)
   - Cred Test ↔ API Enum: 0.74 (similar - both use curl)

4. MODEL PROBING RESULTS
   ┌────────────────────┬──────────────┬───────────────┬──────────┐
   │ Skill              │ Base Model   │ Fine-tuned    │ Δ        │
   ├────────────────────┼──────────────┼───────────────┼──────────┤
   │ Port Enumeration   │ 100%         │ 100%          │ 0%       │
   │ Web Scanning       │ 100%         │ 100%          │ 0%       │
   │ Dir Bruteforce     │ 100%         │ 100%          │ 0%       │
   │ Credential Testing │ 0%           │ 100%          │ +100%    │
   │ API Enumeration    │ 0%           │ 100%          │ +100%    │
   ├────────────────────┼──────────────┼───────────────┼──────────┤
   │ OVERALL            │ 60%          │ 100%          │ +40%     │
   └────────────────────┴──────────────┴───────────────┴──────────┘

5. KEY FINDINGS

   a) SipIt training successfully encodes skill-specific patterns
      - The fine-tuned model correctly differentiates ALL 5 skills
      - Base model fails on non-obvious mappings (creds, API)

   b) Pattern learning happens even from failed chains
      - Only 6% success rate in training data
      - Yet model learns correct tool associations
      - Supports hypothesis: behavioral patterns > success signal

   c) Skills form clusters based on tool usage
      - Port/Web/Dir are distinct (different primary tools)
      - Creds/API overlap (both curl-heavy)
      - This predicts potential confusion between similar skills

   d) First-tool selection is highly stereotyped
      - Port: 100% start with nmap
      - Web: 100% start with nikto
      - Dir: 100% start with gobuster
      - This is "skill fingerprinting" the model has learned

6. INTERPRETABILITY IMPLICATIONS

   The model appears to encode skills as:
   - Tool selection circuits (which tool for which goal)
   - Action sequence templates (stereotyped patterns)
   - NOT deep reasoning (pattern matching dominant)

   This aligns with the SipIt hypothesis:
   - Models learn to imitate successful patterns
   - Domain expertise = internalized action sequences
   - Pattern matching precedes reasoning

================================================================================
"""
    report_path = OUTPUT_DIR / "SUMMARY.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")
    return report


def main():
    """Create all visualizations."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Creating visualizations...")
    create_comparison_chart()
    create_tool_signature_chart()

    print("\nGenerating summary report...")
    report = create_summary_report()
    print(report)


if __name__ == "__main__":
    main()
