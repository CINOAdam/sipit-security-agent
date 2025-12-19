# SipIt Security Agent: Behavioral Consistency Research

## Key Finding

Fine-tuned models can show **100% skill differentiation in probing tests** while exhibiting **0% differentiation in real deployment**. This gap between "what models know" and "what they do" is invisible to standard evaluations.

> **Read the full research journey:** [BLOG_POST.md](BLOG_POST.md)

---

## Mission

Investigate whether LLMs can self-improve on complex pentesting tasks, and develop diagnostic tools to measure behavioral consistency in fine-tuned models.

## Results Summary

| Metric | Value |
|--------|-------|
| Training Chains | 54 across 5 skills |
| Token Accuracy | 99.2% |
| Probe Accuracy | 100% (all skills differentiated) |
| Real Behavior Accuracy | 20% (collapsed to single tool) |
| **Trust Score** | **0.2 (Low — unreliable)** |

## The Trust Diagnostic Framework

Our key contribution: a method to predict deployment reliability before failure.

```python
def trust_score(model, task_type):
    probe_result = probe_skill_selection(model, task_type)
    actual_result = run_agent(model, task_type)
    return similarity(probe_result, actual_result)
```

| Trust Score | Interpretation |
|-------------|----------------|
| > 0.8 | High trust — behavior matches capabilities |
| 0.4 - 0.8 | Medium trust — partial consistency |
| < 0.4 | Low trust — "talks the talk, doesn't walk the walk" |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SECURITY AGENT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   Goal   │───▶│  Decide  │───▶│ Execute  │───▶│ Observe  │  │
│  │          │    │  (LLM)   │    │  (Tools) │    │ (Parse)  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │              │                                │          │
│       │              │         ┌──────────┐          │          │
│       │              └────────▶│  Memory  │◀─────────┘          │
│       │                        │ (History)│                      │
│       │                        └──────────┘                      │
│       │                              │                           │
│       │                              ▼                           │
│       │                     ┌──────────────┐                    │
│       └────────────────────▶│   Verify     │                    │
│                             │  (Success?)  │                    │
│                             └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  PROBE TESTS     │    │  SELF-IMPROVE    │    │  TRUST SCORE     │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ Skill detection  │    │ Collect chains   │    │ Compare probe    │
│ via controlled   │    │ Retrain on       │    │ vs actual        │
│ generation       │    │ successes        │    │ behavior         │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install pydantic-ai transformers peft trl datasets torch

# Collect training chains
python -m security_agent.collect_multiskill --runs 3

# Train multi-skill model
python -m security_agent.train_multiskill --epochs 5

# Probe skill differentiation (controlled test)
python -m security_agent.probe_generation

# Analyze behavioral patterns
python -m security_agent.analyze_chains

# Run self-improvement loop
python -m security_agent.self_improve --iterations 3
```

## Skills Analyzed

| Skill | Signature Tool | Training % | Probe | Real |
|-------|----------------|------------|-------|------|
| Port Enumeration | nmap | 96% | 100% | 100% |
| Web Scanning | nikto | 100% | 100% | 0% |
| Directory Bruteforce | gobuster | 72% | 100% | 0% |
| Credential Testing | nc/curl | 90% | 100% | 0% |
| API Enumeration | curl | 53% | 100% | 0% |

## Key Insights

1. **Probing ≠ Real Behavior** — Models can ace narrow tests while failing deployment
2. **Pattern Matching is Fragile** — Surface patterns don't generalize to full generation
3. **Scale Matters** — 54 chains insufficient for robust behavioral change
4. **Self-Improvement Needs Diversity** — Collapsed behavior prevents exploration

## File Structure

```
security_agent/
├── agent.py                 # PydanticAI security agent
├── tools.py                 # nmap, nikto, gobuster, nc, curl, searchsploit
├── goals.py                 # Goal definitions
├── verification.py          # Success verification
├── collect_multiskill.py    # Multi-skill chain collection
├── train_multiskill.py      # LoRA fine-tuning
├── analyze_chains.py        # Behavioral pattern analysis
├── analyze_skills.py        # Activation analysis (WIP)
├── probe_generation.py      # Skill probing via generation
├── probe_skills.py          # Token-level probing
├── local_agent.py           # Local model inference agent
├── self_improve.py          # Self-improvement loop
├── visualize_results.py     # Visualization generation
├── v2_trainer.py            # Original V2 trainer
├── BLOG_POST.md             # Full research writeup
├── multiskill_chains/       # Collected training data
├── multiskill_model/        # Trained model adapters
├── interpretability_results/ # Analysis outputs
└── self_improve_iterations/ # Self-improvement results
```

## Environment

### Requirements
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM for 7B model)
- Kali Linux VM for tool execution
- Target VMs (DVWA, Juice Shop, Metasploitable2, etc.)

### Environment Variables
```bash
OPENAI_API_KEY=...      # For GPT-4o chain collection
KALI_HOST=...           # Kali VM IP
KALI_USER=...           # SSH user
KALI_KEY_PATH=...       # SSH key path
```

## Citation

```bibtex
@misc{sipit-security-agent-2024,
  title={Behavioral Consistency in Fine-Tuned LLMs: When Probing Doesn't Predict Performance},
  year={2024},
  howpublished={GitHub}
}
```

## License

Research code for educational purposes. Security testing tools should only be used against systems you own or have explicit permission to test.
