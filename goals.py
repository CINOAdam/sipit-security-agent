"""
Goal definitions for security agent.

Each goal represents a pentesting objective with:
- Description of what to achieve
- Success criteria (how to verify completion)
- Target information
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class GoalType(Enum):
    """Types of pentesting goals."""
    GET_SHELL = "get_shell"           # Obtain remote shell access
    FIND_FLAG = "find_flag"           # Find a CTF-style flag
    ENUMERATE = "enumerate"           # Enumerate services/users/shares
    CREDENTIAL = "credential"         # Extract credentials
    PRIVILEGE_ESCALATION = "privesc"  # Escalate privileges


@dataclass
class Target:
    """Target system information."""
    ip: str
    hostname: Optional[str] = None
    known_services: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


@dataclass
class Goal:
    """A pentesting goal for the agent to achieve."""
    id: str
    type: GoalType
    description: str
    target: Target
    success_criteria: str
    hints: List[str] = field(default_factory=list)
    max_steps: int = 20
    timeout_seconds: int = 600

    def to_prompt(self) -> str:
        """Format goal for inclusion in agent prompt."""
        lines = [
            f"# Goal: {self.description}",
            f"Type: {self.type.value}",
            f"Target: {self.target.ip}",
        ]

        if self.target.hostname:
            lines.append(f"Hostname: {self.target.hostname}")

        if self.target.known_services:
            lines.append("Known services:")
            for svc in self.target.known_services:
                lines.append(f"  - {svc}")

        lines.append(f"\nSuccess criteria: {self.success_criteria}")

        if self.hints:
            lines.append("\nHints:")
            for hint in self.hints:
                lines.append(f"  - {hint}")

        lines.append(f"\nMax steps: {self.max_steps}")

        return "\n".join(lines)


# =============================================================================
# Pre-defined Goals for Common Scenarios
# =============================================================================

def create_basic_shell_goal(target_ip: str, target_name: str = "target") -> Goal:
    """Create a basic 'get shell' goal."""
    return Goal(
        id=f"shell_{target_name}",
        type=GoalType.GET_SHELL,
        description=f"Obtain a remote shell on {target_name}",
        target=Target(ip=target_ip, hostname=target_name),
        success_criteria="Establish an interactive shell session on the target",
        hints=[
            "Start with reconnaissance to identify open ports",
            "Look for vulnerable services",
            "Check for known exploits using searchsploit",
        ],
    )


def create_ctf_flag_goal(target_ip: str, flag_format: str = "FLAG{.*}") -> Goal:
    """Create a CTF-style flag finding goal."""
    return Goal(
        id=f"flag_{target_ip.replace('.', '_')}",
        type=GoalType.FIND_FLAG,
        description=f"Find the flag on the target system",
        target=Target(ip=target_ip),
        success_criteria=f"Find a string matching the pattern: {flag_format}",
        hints=[
            "Enumerate the web server for hidden files",
            "Check for SQL injection vulnerabilities",
            "Look in common flag locations (/root/, /home/, web directories)",
        ],
    )


def create_enumeration_goal(target_ip: str, target_info: str = "") -> Goal:
    """Create an enumeration goal."""
    return Goal(
        id=f"enum_{target_ip.replace('.', '_')}",
        type=GoalType.ENUMERATE,
        description=f"Fully enumerate the target system",
        target=Target(ip=target_ip, notes=target_info),
        success_criteria="Identify all open ports, running services, versions, and potential vulnerabilities",
        hints=[
            "Use nmap for port scanning",
            "Run service version detection",
            "Check for web applications with nikto",
            "Search for known vulnerabilities",
        ],
        max_steps=15,
    )


# =============================================================================
# Homelab Target Definitions
# =============================================================================

class HomelabTargets:
    """Pre-defined targets for homelab testing."""

    METASPLOITABLE = Target(
        ip="10.10.10.183",
        hostname="metasploitable",
        known_services=[
            {"port": 21, "service": "vsftpd", "version": "2.3.4"},
            {"port": 22, "service": "OpenSSH"},
            {"port": 23, "service": "Telnet"},
            {"port": 80, "service": "Apache/PHP"},
            {"port": 139, "service": "Samba"},
            {"port": 445, "service": "Samba"},
            {"port": 3306, "service": "MySQL"},
            {"port": 5432, "service": "PostgreSQL"},
            {"port": 6667, "service": "UnrealIRCd", "version": "3.2.8.1"},
            {"port": 8180, "service": "Tomcat"},
        ],
        notes="Classic vulnerable Linux VM (VMID 120)"
    )

    DVWA = Target(
        ip="192.168.0.41",
        hostname="dvwa",
        known_services=[
            {"port": 8001, "service": "http", "note": "DVWA web app"},
        ],
        notes="Damn Vulnerable Web Application (via OPNsense)"
    )

    JUICE_SHOP = Target(
        ip="192.168.0.41",
        hostname="juice-shop",
        known_services=[
            {"port": 8002, "service": "http", "note": "OWASP Juice Shop"},
        ],
        notes="Modern OWASP Top 10 CTF (via OPNsense)"
    )

    VAMPI = Target(
        ip="192.168.0.41",
        hostname="vampi",
        known_services=[
            {"port": 8008, "service": "http", "note": "Vulnerable REST API"},
        ],
        notes="VAmPI - Vulnerable API for testing"
    )

    DVGA = Target(
        ip="192.168.0.41",
        hostname="dvga",
        known_services=[
            {"port": 8007, "service": "http", "note": "GraphQL API"},
        ],
        notes="Damn Vulnerable GraphQL Application"
    )


# =============================================================================
# Goal Library
# =============================================================================

GOAL_LIBRARY = {
    # Easy goals - single known exploit
    "vsftpd_backdoor": Goal(
        id="vsftpd_backdoor",
        type=GoalType.GET_SHELL,
        description="Exploit the vsftpd 2.3.4 backdoor to get a root shell",
        target=HomelabTargets.METASPLOITABLE,
        success_criteria="Obtain shell access via the vsftpd backdoor on port 6200",
        hints=[
            "vsftpd 2.3.4 has a known backdoor triggered by :) in username",
            "The backdoor opens a shell on port 6200",
            "Connect with netcat after triggering",
        ],
        max_steps=10,
    ),

    "unrealircd_backdoor": Goal(
        id="unrealircd_backdoor",
        type=GoalType.GET_SHELL,
        description="Exploit UnrealIRCd 3.2.8.1 backdoor",
        target=HomelabTargets.METASPLOITABLE,
        success_criteria="Obtain shell access via the IRC backdoor",
        hints=[
            "IRC service on port 6667",
            "Backdoor triggered by 'AB;' prefix to command",
        ],
        max_steps=10,
    ),

    "tomcat_shell": Goal(
        id="tomcat_shell",
        type=GoalType.GET_SHELL,
        description="Use Tomcat manager to deploy a shell",
        target=HomelabTargets.METASPLOITABLE,
        success_criteria="Deploy and access a shell via Tomcat manager",
        hints=[
            "Tomcat manager at port 8180",
            "Default credentials tomcat:tomcat",
            "Deploy WAR file with reverse shell",
        ],
        max_steps=15,
    ),

    # Medium goals - require enumeration
    "samba_shell": Goal(
        id="samba_shell",
        type=GoalType.GET_SHELL,
        description="Exploit Samba usermap_script vulnerability",
        target=HomelabTargets.METASPLOITABLE,
        success_criteria="Obtain shell via Samba command injection",
        hints=[
            "SMB on ports 139/445",
            "Username map script allows command injection",
        ],
        max_steps=15,
    ),

    "dvwa_sqli": Goal(
        id="dvwa_sqli",
        type=GoalType.FIND_FLAG,
        description="Extract user credentials via SQL injection in DVWA",
        target=HomelabTargets.DVWA,
        success_criteria="Extract admin password hash from database",
        hints=[
            "Login with admin:password first",
            "SQL injection in user ID field at /vulnerabilities/sqli/",
            "UNION-based extraction",
        ],
        max_steps=20,
    ),

    "dvwa_command": Goal(
        id="dvwa_command",
        type=GoalType.GET_SHELL,
        description="Get shell via command injection in DVWA",
        target=HomelabTargets.DVWA,
        success_criteria="Establish reverse shell via command injection",
        hints=[
            "Command injection in ping utility at /vulnerabilities/exec/",
            "Use semicolon or pipe to chain commands",
        ],
        max_steps=15,
    ),

    # Hard goals - multi-step chains
    "full_enum": Goal(
        id="full_enum",
        type=GoalType.ENUMERATE,
        description="Complete enumeration of Metasploitable2",
        target=HomelabTargets.METASPLOITABLE,
        success_criteria="Identify all 10+ vulnerable services with versions",
        hints=[
            "Use nmap for comprehensive port scan",
            "Run service version detection",
            "Check for known vulnerabilities with searchsploit",
        ],
        max_steps=25,
    ),

    "juice_shop_sqli": Goal(
        id="juice_shop_sqli",
        type=GoalType.CREDENTIAL,
        description="Gain admin access to Juice Shop via SQL injection",
        target=HomelabTargets.JUICE_SHOP,
        success_criteria="Login as admin user",
        hints=[
            "SQL injection in login form",
            "Admin email is admin@juice-sh.op",
        ],
        max_steps=20,
    ),
}


def get_goal(goal_id: str) -> Optional[Goal]:
    """Get a goal by ID."""
    return GOAL_LIBRARY.get(goal_id)


def list_goals() -> List[str]:
    """List all available goal IDs."""
    return list(GOAL_LIBRARY.keys())
