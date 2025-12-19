# Security Agent with SipIt V2 Self-Improvement
#
# A pentesting agent that uses LLM decision-making with reproducibility-based
# self-improvement. Tested against intentionally vulnerable VMs in a homelab.

from .tools import (
    SecurityTool,
    ToolResult,
    ToolCategory,
    get_tool,
    get_all_tools,
    format_tools_for_prompt,
    SECURITY_TOOLS,
)

from .goals import (
    Goal,
    GoalType,
    Target,
    get_goal,
    list_goals,
    create_basic_shell_goal,
    create_enumeration_goal,
    create_ctf_flag_goal,
)

from .verification import (
    GoalVerifier,
    GoalVerification,
    VerificationResult,
    quick_verify,
)

from .agent import (
    SecurityAgent,
    AgentMemory,
    TrainingChain,
    extract_training_chain,
)

__all__ = [
    # Tools
    "SecurityTool",
    "ToolResult",
    "ToolCategory",
    "get_tool",
    "get_all_tools",
    "format_tools_for_prompt",
    "SECURITY_TOOLS",
    # Goals
    "Goal",
    "GoalType",
    "Target",
    "get_goal",
    "list_goals",
    "create_basic_shell_goal",
    "create_enumeration_goal",
    "create_ctf_flag_goal",
    # Verification
    "GoalVerifier",
    "GoalVerification",
    "VerificationResult",
    "quick_verify",
    # Agent
    "SecurityAgent",
    "AgentMemory",
    "TrainingChain",
    "extract_training_chain",
]
