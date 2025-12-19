# When Models Talk the Talk but Don't Walk the Walk: A Journey into LLM Behavioral Consistency

**TL;DR:** We fine-tuned a security agent on 54 task chains, achieved 100% skill differentiation in probing tests, but discovered the model collapsed to a single behavior in real deployment. This gap between "what the model knows" and "what the model does" led us to develop a trust diagnostic framework for evaluating fine-tuned models.

---

## The Starting Point: Can Models Improve Themselves?

This research began with a simple question inspired by the [SipIt paper](https://arxiv.org/abs/2502.01612) (Self-Improvement via Inversion Fidelity Training): **Can we create models that improve through their own experience?**

The vision was compelling:
1. Model attempts tasks
2. Successful attempts become training data
3. Model improves from its own successes
4. Repeat → continuous improvement

We decided to test this in a concrete domain: security/penetration testing agents.

## Building the Security Agent

We built a PydanticAI-based agent with six tools:
- `nmap` — Port and service enumeration
- `nikto` — Web vulnerability scanning
- `gobuster` — Directory bruteforce
- `nc` — Network connections
- `curl` — HTTP requests
- `searchsploit` — Exploit database search

The agent executes against a real homelab environment with vulnerable VMs (DVWA, Juice Shop, Metasploitable2, etc.).

### Multi-Skill Training Data

We collected 54 chains across 5 distinct skill types:

| Skill | Primary Tool | Chains |
|-------|--------------|--------|
| Port Enumeration | nmap (96%) | 12 |
| Web Scanning | nikto (100%) | 12 |
| Directory Bruteforce | gobuster (72%) | 12 |
| Credential Testing | curl/nc (90%) | 9 |
| API Enumeration | curl (53%) | 9 |

Each skill had a clear "behavioral signature" — a dominant tool that characterized that skill type.

### Training Results

We fine-tuned Mistral-7B with LoRA on all 54 chains:
- **Loss:** 2.59 → 0.014
- **Token Accuracy:** 71.1% → 99.2%
- **Training Time:** ~3 minutes

The model learned to reproduce the training patterns with high fidelity.

## The Interpretability Analysis

With a trained model, we wanted to understand: **What did the model actually learn?**

### Chain Analysis

We analyzed the behavioral patterns in our training data:

```
Skill Similarity (based on tool usage):
- Port Enum ↔ Web Scan: 0.04 (highly distinct)
- Dir Brute ↔ API Enum: 0.71 (overlapping)
- Cred Test ↔ API Enum: 0.74 (overlapping)
```

Skills formed natural clusters based on their tool signatures. Port enumeration, web scanning, and directory bruteforce were highly distinctive, while credential testing and API enumeration overlapped (both curl-heavy).

### Probing the Model

We tested whether the fine-tuned model could differentiate skills:

| Skill | Base Model | Fine-tuned | Change |
|-------|------------|------------|--------|
| Port Enumeration | 100% | 100% | — |
| Web Scanning | 100% | 100% | — |
| Directory Bruteforce | 100% | 100% | — |
| Credential Testing | 0% | 100% | **+100%** |
| API Enumeration | 0% | 100% | **+100%** |
| **Overall** | 60% | 100% | **+40%** |

The base model already knew the "obvious" mappings (nmap for ports, nikto for web). But the fine-tuned model learned the non-obvious ones too (nc/curl for credentials, curl for API).

**Success!** Or so we thought...

## The Self-Improvement Experiment

With skill differentiation working, we built the self-improvement loop:

```python
for iteration in range(n):
    chains = collect_with_model(current_model)
    successful = filter_successful(chains)
    current_model = train_on(successful)
```

### The Unexpected Result

When we ran the model in the actual agent loop, something strange happened:

| Goal Type | Expected Tool | Actual Tool | Correct? |
|-----------|---------------|-------------|----------|
| Port Enumeration | nmap | nmap | ✓ |
| Web Scanning | nikto | nmap | ✗ |
| Directory Bruteforce | gobuster | nmap | ✗ |
| Credential Testing | nc/curl | nmap | ✗ |
| API Enumeration | curl | nmap | ✗ |

**The model used nmap for everything.**

Despite 100% skill differentiation in probing, the model collapsed to a single behavior in real deployment. The self-improvement loop was stuck — the model couldn't generate diverse successful chains because it always did the same thing.

## The Key Insight: Probing ≠ Real Behavior

This revealed a critical gap:

```
Probing (controlled, single-token prediction)
    → 100% skill differentiation

Real Behavior (multi-turn generation)
    → 0% skill differentiation (always nmap)
```

The model learned a **superficial pattern**: when it sees a skill label, it can output the corresponding tool token. But it didn't learn to **reason** about tool selection in a generative context.

This is the difference between:
- **Pattern matching:** "skill2_web_scan" → output "nikto"
- **Reasoning:** "I need to find web vulnerabilities, nikto is a web vulnerability scanner, therefore I should use nikto"

Our 54 training chains weren't enough to instill robust behavioral change. The model memorized the surface pattern without internalizing the deeper logic.

## The Trust Diagnostic Framework

This failure mode led us to a valuable tool: **behavioral consistency checking**.

### The Core Idea

```
Trust = Consistency(Probe Response, Actual Behavior)
```

If a model's probed capabilities match its deployed behavior, you can trust it. If there's a gap, the model is unreliable.

### Implementation

```python
def trust_score(model, task_type):
    # What does the model "know"?
    probe_result = probe_skill_selection(model, task_type)

    # What does the model "do"?
    actual_result = run_agent(model, task_type)

    # How consistent are they?
    return similarity(probe_result, actual_result)
```

### Interpretation

| Trust Score | Meaning |
|-------------|---------|
| High (>0.8) | Reliable — internalized patterns match behavior |
| Medium (0.4-0.8) | Caution — partial consistency |
| Low (<0.4) | Unreliable — "talks the talk but doesn't walk the walk" |

Our experiment scored **~0.2** — very low trust. The model appeared capable in testing but failed in deployment.

## Lessons Learned

### 1. Scale Matters More Than We Expected

54 chains wasn't enough for robust behavioral learning. Genuine skill acquisition likely requires:
- 1000s of examples per skill
- Multiple variations of each pattern
- Diverse failure modes to learn from

### 2. Probing is Necessary but Not Sufficient

Probing tests are valuable for understanding what patterns a model has learned. But they don't predict real-world performance. Always validate with end-to-end testing.

### 3. Pattern Matching Precedes Reasoning

Models learn superficial patterns before deep reasoning. Our model learned "skill → tool" mappings but not "why this tool for this task." This is consistent with findings in mechanistic interpretability.

### 4. Self-Improvement Needs Diversity

A self-improvement loop only works if the model can generate diverse behaviors. If it collapses to a single pattern, it can't explore and improve. Breaking out of local optima requires either:
- Much more training data
- Explicit exploration mechanisms
- External reward signals (RL)

## Future Directions

### 1. Trust Diagnostic Tool

Formalize the behavioral consistency framework into a reusable tool for evaluating fine-tuned models before deployment.

### 2. Curriculum Learning

Start with simple, high-success-rate tasks and progressively add complexity. Let the model build a foundation before attempting harder skills.

### 3. Reinforcement Learning

Add actual task success as a reward signal, not just pattern imitation. RL could help the model learn *why* certain tools work, not just *that* they correlate with certain labels.

### 4. Larger Scale

Test whether the self-improvement loop works with 10x or 100x more data. There may be a phase transition where the model goes from pattern matching to genuine reasoning.

## Conclusion

We set out to build a self-improving security agent. We ended up discovering something more fundamental: **the gap between what models "know" and what they "do."**

This gap is invisible in standard evaluations. A model can ace probing tests while completely failing in deployment. The only way to catch this is to test end-to-end behavior and compare it to probed capabilities.

Our trust diagnostic framework turns this failure mode into a feature. By measuring behavioral consistency, we can predict which fine-tuned models will actually work in production — before they fail on real tasks.

The journey from "self-improvement" to "trust diagnostics" wasn't what we planned, but it's arguably more valuable. Self-improvement remains an open problem. But knowing when to trust your model? That's something we can measure today.

---

## Repository Structure

```
security_agent/
├── agent.py                 # PydanticAI security agent
├── tools.py                 # Tool implementations (nmap, nikto, etc.)
├── goals.py                 # Goal definitions
├── verification.py          # Success verification
├── collect_multiskill.py    # Multi-skill chain collection
├── train_multiskill.py      # Multi-skill training
├── analyze_chains.py        # Behavioral pattern analysis
├── probe_generation.py      # Skill probing via generation
├── local_agent.py           # Local model inference agent
├── self_improve.py          # Self-improvement loop
├── visualize_results.py     # Result visualization
├── multiskill_chains/       # Collected training data
├── multiskill_model/        # Trained model adapters
├── interpretability_results/ # Analysis outputs
└── BLOG_POST.md             # This document
```

## Citation

If you use this work, please cite:

```
@misc{sipit-security-agent-2025,
  title={Behavioral Consistency in Fine-Tuned LLMs: When Probing Doesn't Predict Performance},
  author={[Adam Kruger]},
  year={2025},
  howpublished={GitHub}
}
```

## Acknowledgments

- Inspired by the SipIt paper (Self-Improvement via Inversion Fidelity Training)
- Built with PydanticAI, Hugging Face Transformers, and PEFT
- Tested against OWASP vulnerable applications (DVWA, Juice Shop, bWAPP, etc.)
