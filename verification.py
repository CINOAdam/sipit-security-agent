"""
Success verification for security agent goals.

Provides binary verification of goal completion:
- Shell detection (did we get a shell?)
- Flag extraction (did we find the flag?)
- Enumeration completeness (did we find all services?)
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

from .goals import Goal, GoalType
from .tools import ToolResult


class VerificationResult(Enum):
    """Result of goal verification."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class GoalVerification:
    """Result of verifying a goal."""
    result: VerificationResult
    confidence: float  # 0.0 to 1.0
    evidence: str
    details: Dict[str, Any]


class GoalVerifier:
    """Verifies if goals have been achieved."""

    def __init__(self):
        self.shell_indicators = [
            r"\$\s*$",           # Shell prompt $
            r"#\s*$",            # Root shell prompt #
            r"root@",            # Root user
            r"uid=\d+",          # id command output
            r"Linux.*\d+\.\d+",  # uname output
            r"bash-\d+",         # Bash version
            r"sh-\d+",           # Shell version
        ]

        self.flag_patterns = [
            r"FLAG\{[^}]+\}",
            r"flag\{[^}]+\}",
            r"CTF\{[^}]+\}",
            r"[a-f0-9]{32}",     # MD5-like hash as flag
        ]

    def verify(self, goal: Goal, history: List[ToolResult]) -> GoalVerification:
        """Verify if a goal has been achieved based on tool history."""
        if goal.type == GoalType.GET_SHELL:
            return self._verify_shell(goal, history)
        elif goal.type == GoalType.FIND_FLAG:
            return self._verify_flag(goal, history)
        elif goal.type == GoalType.ENUMERATE:
            return self._verify_enumeration(goal, history)
        elif goal.type == GoalType.CREDENTIAL:
            return self._verify_credentials(goal, history)
        else:
            return GoalVerification(
                result=VerificationResult.UNKNOWN,
                confidence=0.0,
                evidence="Unknown goal type",
                details={},
            )

    def _verify_shell(self, goal: Goal, history: List[ToolResult]) -> GoalVerification:
        """Verify if shell access was obtained."""
        evidence_lines = []
        shell_score = 0.0

        for result in history:
            output = result.output

            # Check for shell indicators
            for pattern in self.shell_indicators:
                if re.search(pattern, output, re.MULTILINE):
                    evidence_lines.append(f"Found shell indicator: {pattern}")
                    shell_score += 0.2

            # Check for interactive shell signs
            if "whoami" in result.command.lower() and result.success:
                evidence_lines.append("Successfully ran whoami")
                shell_score += 0.3

            if "id" in result.command.lower() and "uid=" in output:
                evidence_lines.append("Successfully ran id command")
                shell_score += 0.3

            # Netcat listener received connection
            if result.tool == "nc" and "connect" in output.lower():
                evidence_lines.append("Netcat received connection")
                shell_score += 0.4

            # Check for reverse shell connection
            if "Connection received" in output or "connect to" in output.lower():
                evidence_lines.append("Reverse shell connection detected")
                shell_score += 0.5

        shell_score = min(1.0, shell_score)

        if shell_score >= 0.7:
            result = VerificationResult.SUCCESS
        elif shell_score >= 0.3:
            result = VerificationResult.PARTIAL
        else:
            result = VerificationResult.FAILURE

        return GoalVerification(
            result=result,
            confidence=shell_score,
            evidence="; ".join(evidence_lines) if evidence_lines else "No shell evidence found",
            details={"indicators_found": len(evidence_lines)},
        )

    def _verify_flag(self, goal: Goal, history: List[ToolResult]) -> GoalVerification:
        """Verify if a flag was found."""
        found_flags = []

        for result in history:
            output = result.output

            for pattern in self.flag_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                found_flags.extend(matches)

        if found_flags:
            return GoalVerification(
                result=VerificationResult.SUCCESS,
                confidence=1.0,
                evidence=f"Found flag(s): {found_flags}",
                details={"flags": found_flags},
            )
        else:
            return GoalVerification(
                result=VerificationResult.FAILURE,
                confidence=0.0,
                evidence="No flags found in output",
                details={"flags": []},
            )

    def _verify_enumeration(self, goal: Goal, history: List[ToolResult]) -> GoalVerification:
        """Verify enumeration completeness."""
        findings = {
            "ports_scanned": False,
            "services_identified": False,
            "versions_detected": False,
            "vulnerabilities_checked": False,
            "web_enumerated": False,
        }

        ports_found = []
        services_found = []

        for result in history:
            # Check for port scanning
            if result.tool == "nmap":
                findings["ports_scanned"] = True
                if result.parsed_data and result.parsed_data.get("open_ports"):
                    ports_found.extend(result.parsed_data["open_ports"])
                    findings["services_identified"] = True
                if "-sV" in result.command or "version" in result.output.lower():
                    findings["versions_detected"] = True

            # Check for vulnerability scanning
            if result.tool == "nikto":
                findings["vulnerabilities_checked"] = True
                findings["web_enumerated"] = True

            # Check for directory enumeration
            if result.tool == "gobuster":
                findings["web_enumerated"] = True

            # Check for exploit search
            if result.tool == "searchsploit":
                findings["vulnerabilities_checked"] = True

        # Calculate score
        completed = sum(1 for v in findings.values() if v)
        total = len(findings)
        score = completed / total

        if score >= 0.8:
            result = VerificationResult.SUCCESS
        elif score >= 0.4:
            result = VerificationResult.PARTIAL
        else:
            result = VerificationResult.FAILURE

        return GoalVerification(
            result=result,
            confidence=score,
            evidence=f"Completed {completed}/{total} enumeration tasks",
            details={
                "findings": findings,
                "ports": ports_found,
                "services": services_found,
            },
        )

    def _verify_credentials(self, goal: Goal, history: List[ToolResult]) -> GoalVerification:
        """Verify if credentials were extracted."""
        cred_patterns = [
            r"username[:\s]+(\w+)",
            r"password[:\s]+(\S+)",
            r"(\w+):(\S+)@",  # user:pass@host
            r"(\w+):(\S+)",   # user:pass in password file
        ]

        found_creds = []

        for result in history:
            output = result.output
            for pattern in cred_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    found_creds.extend(matches)

        if found_creds:
            return GoalVerification(
                result=VerificationResult.SUCCESS,
                confidence=0.8,
                evidence=f"Found {len(found_creds)} potential credential(s)",
                details={"credentials_count": len(found_creds)},
            )
        else:
            return GoalVerification(
                result=VerificationResult.FAILURE,
                confidence=0.0,
                evidence="No credentials found",
                details={"credentials_count": 0},
            )


# =============================================================================
# Utility Functions
# =============================================================================

def quick_verify(goal: Goal, history: List[ToolResult]) -> bool:
    """Quick binary check if goal was achieved."""
    verifier = GoalVerifier()
    result = verifier.verify(goal, history)
    return result.result == VerificationResult.SUCCESS
