#!/usr/bin/env python3
"""
Baseline evaluation for security agent.

Measures success rate on pentesting goals before V2 training.
"""

import json
import asyncio
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .agent import SecurityAgent, AgentMemory, TrainingChain, extract_training_chain
from .goals import Goal, GoalType, Target, create_basic_shell_goal, create_enumeration_goal
from .verification import GoalVerifier, VerificationResult


@dataclass
class EvaluationResult:
    """Result of evaluating on a single goal."""
    goal_id: str
    goal_description: str
    target_ip: str
    success: bool
    steps_taken: int
    duration_seconds: float
    verification_confidence: float
    chain: Optional[TrainingChain] = None
    error: Optional[str] = None


@dataclass
class BaselineStats:
    """Aggregate statistics from baseline evaluation."""
    total_goals: int
    successful: int
    failed: int
    success_rate: float
    avg_steps: float
    avg_duration: float
    by_difficulty: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_targets_config(config_path: str = None) -> Dict[str, Any]:
    """Load targets configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "targets.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_goals_from_config(config: Dict[str, Any]) -> List[tuple]:
    """Create Goal objects from config, returns (goal, difficulty) tuples."""
    goals = []
    targets = config.get("targets", {})
    eval_goals = config.get("evaluation_goals", [])

    for goal_def in eval_goals:
        target_name = goal_def.get("target")
        target_config = targets.get(target_name, {})

        target = Target(
            ip=target_config.get("ip", "127.0.0.1"),
            hostname=target_config.get("hostname", target_name),
            notes=target_config.get("description", ""),
        )

        goal_type = {
            "get_shell": GoalType.GET_SHELL,
            "find_flag": GoalType.FIND_FLAG,
            "enumerate": GoalType.ENUMERATE,
            "credential": GoalType.CREDENTIAL,
        }.get(goal_def.get("type"), GoalType.GET_SHELL)

        goal = Goal(
            id=goal_def.get("id"),
            type=goal_type,
            description=goal_def.get("description"),
            target=target,
            success_criteria=f"Achieve {goal_def.get('type')} on {target_name}",
            max_steps=30,
        )

        difficulty = goal_def.get("difficulty", "medium")
        goals.append((goal, difficulty))

    return goals


class BaselineEvaluator:
    """Runs baseline evaluation on pentesting goals."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.agent = SecurityAgent(
            model_name=model,
            base_url=base_url,
            api_key=api_key,
        )
        self.verifier = GoalVerifier()
        self.results: List[EvaluationResult] = []

    async def evaluate_goal(
        self,
        goal: Goal,
        runs: int = 1,
        verbose: bool = True,
    ) -> List[EvaluationResult]:
        """Evaluate agent on a single goal with multiple runs."""
        results = []

        for run_idx in range(runs):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Goal: {goal.id} (Run {run_idx + 1}/{runs})")
                print(f"{'='*60}")

            try:
                start_time = datetime.now()
                memory = await self.agent.run(goal, verbose=verbose)
                end_time = datetime.now()

                duration = (end_time - start_time).total_seconds()

                verification = self.verifier.verify(goal, memory.tool_results)
                chain = extract_training_chain(memory, verification)

                result = EvaluationResult(
                    goal_id=goal.id,
                    goal_description=goal.description,
                    target_ip=goal.target.ip,
                    success=verification.result == VerificationResult.SUCCESS,
                    steps_taken=len(chain.actions),
                    duration_seconds=duration,
                    verification_confidence=verification.confidence,
                    chain=chain,
                )

            except Exception as e:
                result = EvaluationResult(
                    goal_id=goal.id,
                    goal_description=goal.description,
                    target_ip=goal.target.ip,
                    success=False,
                    steps_taken=0,
                    duration_seconds=0.0,
                    verification_confidence=0.0,
                    error=str(e),
                )

            results.append(result)
            self.results.append(result)

            if verbose:
                status = "SUCCESS" if result.success else "FAILED"
                print(f"\nResult: {status}")
                print(f"Steps: {result.steps_taken}")
                print(f"Duration: {result.duration_seconds:.1f}s")
                if result.error:
                    print(f"Error: {result.error}")

        return results

    async def run_baseline(
        self,
        goals: List[tuple],  # (Goal, difficulty) tuples
        runs_per_goal: int = 1,
        verbose: bool = True,
    ) -> BaselineStats:
        """Run full baseline evaluation."""
        if verbose:
            print("\n" + "="*70)
            print("SECURITY AGENT BASELINE EVALUATION")
            print("="*70)
            print(f"Goals: {len(goals)}")
            print(f"Runs per goal: {runs_per_goal}")
            print("="*70)

        by_difficulty = {}

        for goal, difficulty in goals:
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {
                    "total": 0,
                    "successful": 0,
                    "total_steps": 0,
                    "total_duration": 0.0,
                }

            results = await self.evaluate_goal(goal, runs_per_goal, verbose)

            for result in results:
                by_difficulty[difficulty]["total"] += 1
                if result.success:
                    by_difficulty[difficulty]["successful"] += 1
                by_difficulty[difficulty]["total_steps"] += result.steps_taken
                by_difficulty[difficulty]["total_duration"] += result.duration_seconds

        # Calculate aggregate stats
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        total_steps = sum(r.steps_taken for r in self.results)
        total_duration = sum(r.duration_seconds for r in self.results)

        stats = BaselineStats(
            total_goals=total,
            successful=successful,
            failed=total - successful,
            success_rate=successful / total if total > 0 else 0.0,
            avg_steps=total_steps / total if total > 0 else 0.0,
            avg_duration=total_duration / total if total > 0 else 0.0,
            by_difficulty={
                diff: {
                    "success_rate": data["successful"] / data["total"] if data["total"] > 0 else 0.0,
                    "avg_steps": data["total_steps"] / data["total"] if data["total"] > 0 else 0.0,
                    "avg_duration": data["total_duration"] / data["total"] if data["total"] > 0 else 0.0,
                }
                for diff, data in by_difficulty.items()
            },
        )

        if verbose:
            self._print_stats(stats)

        return stats

    def _print_stats(self, stats: BaselineStats):
        """Print statistics."""
        print("\n" + "="*70)
        print("BASELINE RESULTS")
        print("="*70)

        print(f"\nOverall:")
        print(f"  Total goals: {stats.total_goals}")
        print(f"  Successful: {stats.successful}")
        print(f"  Failed: {stats.failed}")
        print(f"  Success rate: {stats.success_rate:.1%}")
        print(f"  Avg steps: {stats.avg_steps:.1f}")
        print(f"  Avg duration: {stats.avg_duration:.1f}s")

        if stats.by_difficulty:
            print("\nBy Difficulty:")
            for diff, data in sorted(stats.by_difficulty.items()):
                print(f"  {diff}:")
                print(f"    Success rate: {data['success_rate']:.1%}")
                print(f"    Avg steps: {data['avg_steps']:.1f}")
                print(f"    Avg duration: {data['avg_duration']:.1f}s")

    def save_results(self, output_path: str):
        """Save results to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "goal_id": r.goal_id,
                    "goal_description": r.goal_description,
                    "target_ip": r.target_ip,
                    "success": r.success,
                    "steps_taken": r.steps_taken,
                    "duration_seconds": r.duration_seconds,
                    "verification_confidence": r.verification_confidence,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {output_path}")


# =============================================================================
# Simulated Baseline (for testing without real targets)
# =============================================================================

async def run_simulated_baseline(
    model: str = "gpt-4o-mini",
    num_goals: int = 5,
    verbose: bool = True,
):
    """
    Run a simulated baseline that doesn't require real targets.

    Uses placeholder targets to test agent decision-making.
    """
    # Create simulated goals
    goals = [
        (create_basic_shell_goal("192.168.1.100", "target1"), "easy"),
        (create_basic_shell_goal("192.168.1.101", "target2"), "easy"),
        (create_enumeration_goal("192.168.1.102", "Web server"), "medium"),
    ][:num_goals]

    evaluator = BaselineEvaluator(model=model)

    # Note: This won't actually succeed since tools won't connect
    # It's for testing the agent's decision-making logic
    stats = await evaluator.run_baseline(goals, runs_per_goal=1, verbose=verbose)

    evaluator.save_results("simulated_baseline.json")

    return stats


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Security Agent Baseline Evaluation")
    parser.add_argument("--config", "-c", help="Path to targets.yaml")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Runs per goal")
    parser.add_argument("--output", "-o", default="baseline_results.json", help="Output file")
    parser.add_argument("--simulated", "-s", action="store_true", help="Run simulated baseline")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    if args.simulated:
        await run_simulated_baseline(model=args.model, verbose=not args.quiet)
        return

    # Load config
    config = load_targets_config(args.config)
    if not config:
        print("No config found. Use --simulated for testing without targets.")
        return

    goals = create_goals_from_config(config)
    if not goals:
        print("No goals found in config.")
        return

    evaluator = BaselineEvaluator(model=args.model)
    stats = await evaluator.run_baseline(goals, runs_per_goal=args.runs, verbose=not args.quiet)
    evaluator.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())
