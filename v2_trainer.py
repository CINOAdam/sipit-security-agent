"""
SipIt V2 Training for Security Agent Chains.

Adapts the reproducibility + fidelity approach to agent action chains:
- Reproducibility: Do similar tool sequences emerge for the same goal?
- Fidelity: Did the chain actually achieve the goal? (binary, execution-verified)

Only trains on chains that are both successful AND reproducible.
"""

import json
import asyncio
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Load .env
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from .agent import SecurityAgent, AgentMemory, TrainingChain, extract_training_chain
from .goals import Goal, get_goal, list_goals
from .verification import GoalVerifier, VerificationResult


# =============================================================================
# Chain Similarity
# =============================================================================

@dataclass
class ChainSignature:
    """Normalized representation of an action chain for comparison."""
    goal_id: str
    tool_sequence: List[str]  # Just tool names in order
    key_parameters: Dict[str, Any]  # Important parameters
    success: bool

    def to_string(self) -> str:
        """Convert to string for embedding."""
        tools = " -> ".join(self.tool_sequence)
        return f"Goal: {self.goal_id}\nTools: {tools}\nSuccess: {self.success}"


def extract_signature(chain: TrainingChain) -> ChainSignature:
    """Extract a normalized signature from a chain."""
    tool_sequence = [a["tool"] for a in chain.actions]

    # Extract key parameters (target-related)
    key_params = {}
    for action in chain.actions:
        if "target" in action["parameters"]:
            key_params["target"] = action["parameters"]["target"]
            break

    return ChainSignature(
        goal_id=chain.goal_id,
        tool_sequence=tool_sequence,
        key_parameters=key_params,
        success=chain.success,
    )


class ChainSimilarityScorer:
    """Computes similarity between action chains."""

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(embed_model)

    def semantic_similarity(self, chain1: TrainingChain, chain2: TrainingChain) -> float:
        """Compute semantic similarity between two chains."""
        sig1 = extract_signature(chain1)
        sig2 = extract_signature(chain2)

        text1 = sig1.to_string()
        text2 = sig2.to_string()

        embeddings = self.embed_model.encode([text1, text2])
        similarity = float(F.cosine_similarity(
            torch.tensor(embeddings[0]).unsqueeze(0),
            torch.tensor(embeddings[1]).unsqueeze(0)
        ).item())

        return (similarity + 1) / 2  # Normalize to 0-1

    def tool_overlap(self, chain1: TrainingChain, chain2: TrainingChain) -> float:
        """Compute tool sequence overlap."""
        tools1 = set(a["tool"] for a in chain1.actions)
        tools2 = set(a["tool"] for a in chain2.actions)

        if not tools1 or not tools2:
            return 0.0

        intersection = len(tools1 & tools2)
        union = len(tools1 | tools2)

        return intersection / union if union > 0 else 0.0

    def combined_similarity(self, chain1: TrainingChain, chain2: TrainingChain) -> float:
        """Combined similarity score."""
        semantic = self.semantic_similarity(chain1, chain2)
        overlap = self.tool_overlap(chain1, chain2)
        return 0.6 * semantic + 0.4 * overlap


# =============================================================================
# V2 Chain Collector
# =============================================================================

@dataclass
class ChainCollection:
    """Collection of chains for a specific goal."""
    goal_id: str
    chains: List[TrainingChain] = field(default_factory=list)
    successful_chains: List[TrainingChain] = field(default_factory=list)

    def add(self, chain: TrainingChain):
        self.chains.append(chain)
        if chain.success:
            self.successful_chains.append(chain)

    @property
    def success_rate(self) -> float:
        if not self.chains:
            return 0.0
        return len(self.successful_chains) / len(self.chains)


class V2ChainCollector:
    """
    Collects and filters chains using V2 criteria.

    For each goal:
    1. Run the agent N times
    2. Filter for successful chains (fidelity)
    3. Check reproducibility among successful chains
    4. Keep only chains that are both successful AND reproducible
    """

    def __init__(
        self,
        agent: SecurityAgent,
        similarity_threshold: float = 0.6,
        min_reproducibility: int = 2,  # Need at least this many similar chains
    ):
        self.agent = agent
        self.verifier = GoalVerifier()
        self.similarity_scorer = ChainSimilarityScorer()
        self.similarity_threshold = similarity_threshold
        self.min_reproducibility = min_reproducibility

        self.collections: Dict[str, ChainCollection] = {}

    async def collect_for_goal(
        self,
        goal: Goal,
        num_runs: int = 5,
        verbose: bool = True,
    ) -> ChainCollection:
        """Collect chains for a single goal."""
        collection = ChainCollection(goal_id=goal.id)

        if verbose:
            print(f"\nCollecting {num_runs} chains for goal: {goal.id}")

        for run_idx in range(num_runs):
            if verbose:
                print(f"\n  Run {run_idx + 1}/{num_runs}...")

            try:
                memory = await self.agent.run(goal, verbose=False)
                verification = self.verifier.verify(goal, memory.tool_results)
                chain = extract_training_chain(memory, verification)
                collection.add(chain)

                if verbose:
                    status = "SUCCESS" if chain.success else "FAILED"
                    print(f"    Result: {status} ({len(chain.actions)} steps)")

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")

        self.collections[goal.id] = collection

        if verbose:
            print(f"\n  Summary: {len(collection.successful_chains)}/{len(collection.chains)} successful")

        return collection

    def filter_reproducible(
        self,
        collection: ChainCollection,
        verbose: bool = True,
    ) -> List[TrainingChain]:
        """Filter for reproducible chains among successful ones."""
        if len(collection.successful_chains) < self.min_reproducibility:
            if verbose:
                print(f"  Not enough successful chains for reproducibility check")
            return []

        reproducible = []

        for chain in collection.successful_chains:
            # Count how many other chains are similar
            similar_count = 0
            for other in collection.successful_chains:
                if chain is not other:
                    similarity = self.similarity_scorer.combined_similarity(chain, other)
                    if similarity >= self.similarity_threshold:
                        similar_count += 1

            # Need at least min_reproducibility-1 similar chains (excluding self)
            if similar_count >= self.min_reproducibility - 1:
                reproducible.append(chain)
                if verbose:
                    print(f"    Chain with {len(chain.actions)} steps is reproducible "
                          f"({similar_count + 1} similar chains)")

        return reproducible

    async def collect_training_data(
        self,
        goals: List[Goal],
        runs_per_goal: int = 5,
        verbose: bool = True,
    ) -> List[TrainingChain]:
        """Collect training data across multiple goals."""
        all_training_chains = []

        for goal in goals:
            collection = await self.collect_for_goal(goal, runs_per_goal, verbose)
            reproducible = self.filter_reproducible(collection, verbose)
            all_training_chains.extend(reproducible)

            if verbose:
                print(f"\n  Accepted {len(reproducible)} reproducible chains for {goal.id}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Total training chains: {len(all_training_chains)}")
            print(f"{'='*70}")

        return all_training_chains


# =============================================================================
# Chain-to-Training Format
# =============================================================================

def chain_to_training_example(chain: TrainingChain) -> Dict[str, str]:
    """Convert a chain to training format."""
    # Format as a multi-turn conversation
    messages = []

    # System message with goal
    system = f"""You are an expert penetration tester.
Goal: {chain.goal_description}
Target: {chain.target_ip}

Select and execute tools to achieve the goal."""

    messages.append({"role": "system", "content": system})

    # Each action becomes a turn
    for i, action in enumerate(chain.actions):
        # Format the action as a tool call
        tool_call = {
            "name": action["tool"],
            "parameters": action["parameters"],
        }
        messages.append({
            "role": "assistant",
            "content": json.dumps(tool_call),
        })

        # Simulate tool result
        result_msg = f"[Tool executed: {action['tool']}]"
        if action["success"]:
            result_msg += " Success."
        else:
            result_msg += " Failed."
        messages.append({
            "role": "user",
            "content": result_msg,
        })

    return {
        "messages": messages,
        "goal_id": chain.goal_id,
        "success": chain.success,
        "num_steps": len(chain.actions),
    }


# =============================================================================
# V2 Trainer for Security Agent
# =============================================================================

class V2SecurityTrainer:
    """
    Trains the security agent using SipIt V2 approach.

    1. Collect chains with reproducibility filtering
    2. Convert to training format
    3. Fine-tune with LoRA
    """

    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda",
        output_dir: str = "./security_agent_v2",
    ):
        self.base_model = base_model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer
        print(f"Loading model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train_on_chains(
        self,
        chains: List[TrainingChain],
        epochs: int = 3,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
    ):
        """Train on collected chains."""
        if not chains:
            print("No chains to train on!")
            return

        print(f"\nTraining on {len(chains)} chains...")

        # Convert to training format
        examples = [chain_to_training_example(c) for c in chains]

        # Tokenize
        tokenized = []
        for ex in examples:
            # Format as chat
            text = self.tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            tokenized.append(tokens)

        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()
        self.model.gradient_checkpointing_enable()

        total_steps = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for i, tokens in enumerate(tokenized):
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                epoch_loss += loss.item() * gradient_accumulation_steps

                if (i + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_steps += 1

            # Handle remaining gradients
            if len(tokenized) % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                total_steps += 1

            avg_loss = epoch_loss / len(tokenized)
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

        # Save
        save_path = self.output_dir / "final_model"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\nModel saved to {save_path}")

    def save_chains(self, chains: List[TrainingChain], filename: str = "training_chains.json"):
        """Save chains to file."""
        path = self.output_dir / filename
        data = []
        for chain in chains:
            data.append({
                "goal_id": chain.goal_id,
                "goal_description": chain.goal_description,
                "target_ip": chain.target_ip,
                "actions": chain.actions,
                "success": chain.success,
                "duration_seconds": chain.duration_seconds,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Chains saved to {path}")


# =============================================================================
# Main Training Pipeline
# =============================================================================

async def run_v2_training(
    goals: Optional[List[str]] = None,
    runs_per_goal: int = 5,
    model: str = "gpt-4o-mini",
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "./security_agent_v2",
    verbose: bool = True,
):
    """Run the full V2 training pipeline."""
    # Get goals
    if goals is None:
        goals = list_goals()

    goal_objects = [get_goal(g) for g in goals if get_goal(g)]

    if not goal_objects:
        print("No valid goals found!")
        return

    print(f"Training on {len(goal_objects)} goals")

    # Create agent for data collection
    agent = SecurityAgent(model_name=model)

    # Collect training data
    collector = V2ChainCollector(agent)
    training_chains = await collector.collect_training_data(
        goal_objects,
        runs_per_goal=runs_per_goal,
        verbose=verbose,
    )

    if not training_chains:
        print("\nNo reproducible successful chains found. Need more runs or easier goals.")
        return

    # Train
    trainer = V2SecurityTrainer(
        base_model=base_model,
        output_dir=output_dir,
    )
    trainer.save_chains(training_chains)
    trainer.train_on_chains(training_chains)

    print("\nV2 training complete!")


if __name__ == "__main__":
    asyncio.run(run_v2_training(verbose=True))
