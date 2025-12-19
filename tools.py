"""
Security tools for the pentesting agent.

Each tool wraps a real security tool and provides:
- Schema definition (for the LLM)
- Execution function (actually runs the tool)
- Output parsing (normalizes results)
"""

import subprocess
import shlex
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import os

# Remote execution config (Kali box)
REMOTE_HOST = os.environ.get("PENTEST_HOST", "192.168.0.225")
REMOTE_USER = os.environ.get("PENTEST_USER", "netdash")
REMOTE_PASS = os.environ.get("PENTEST_PASS", "netdash")
USE_REMOTE = os.environ.get("USE_REMOTE_EXEC", "true").lower() == "true"


class ToolCategory(Enum):
    RECON = "reconnaissance"
    SCANNING = "scanning"
    ENUMERATION = "enumeration"
    EXPLOITATION = "exploitation"
    POST_EXPLOIT = "post_exploitation"
    UTILITY = "utility"


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool: str
    command: str
    success: bool
    output: str
    parsed_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class SecurityTool:
    """Definition of a security tool."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    examples: List[str]

    def build_command(self, **kwargs) -> str:
        """Build the command string from parameters. Override in subclasses."""
        raise NotImplementedError

    def parse_output(self, output: str) -> Dict[str, Any]:
        """Parse tool output into structured data. Override in subclasses."""
        return {"raw": output}

    async def execute(self, timeout: int = 300, **kwargs) -> ToolResult:
        """Execute the tool and return results."""
        try:
            command = self.build_command(**kwargs)

            # Execute remotely via SSH if configured
            if USE_REMOTE:
                ssh_cmd = f"sshpass -p '{REMOTE_PASS}' ssh -o StrictHostKeyChecking=no {REMOTE_USER}@{REMOTE_HOST} {shlex.quote(command)}"
                proc = await asyncio.create_subprocess_shell(
                    ssh_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(
                    tool=self.name,
                    command=command,
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout}s"
                )

            output = stdout.decode('utf-8', errors='ignore')
            error = stderr.decode('utf-8', errors='ignore')

            parsed = self.parse_output(output)

            return ToolResult(
                tool=self.name,
                command=command,
                success=proc.returncode == 0,
                output=output,
                parsed_data=parsed,
                error=error if error else None,
            )

        except Exception as e:
            return ToolResult(
                tool=self.name,
                command=kwargs.get('_raw_command', str(kwargs)),
                success=False,
                output="",
                error=str(e),
            )


# =============================================================================
# Tool Implementations
# =============================================================================

class NmapTool(SecurityTool):
    """Network scanner and port discovery."""

    def __init__(self):
        super().__init__(
            name="nmap",
            description="Network scanner for host discovery and port scanning",
            category=ToolCategory.SCANNING,
            parameters={
                "target": {"type": "string", "required": True, "description": "Target IP or hostname"},
                "ports": {"type": "string", "required": False, "description": "Port range (e.g., '1-1000', '22,80,443')"},
                "scan_type": {"type": "string", "required": False, "enum": ["syn", "tcp", "udp", "ping"], "description": "Scan type"},
                "scripts": {"type": "string", "required": False, "description": "NSE scripts to run"},
                "os_detect": {"type": "boolean", "required": False, "description": "Enable OS detection"},
                "version_detect": {"type": "boolean", "required": False, "description": "Enable version detection"},
            },
            examples=[
                "nmap -sS -p 1-1000 192.168.1.100",
                "nmap -sV -sC 10.0.0.1",
                "nmap -O -A 192.168.1.0/24",
            ]
        )

    def build_command(self, target: str, ports: str = None, scan_type: str = None,
                      scripts: str = None, os_detect: bool = False,
                      version_detect: bool = False, **kwargs) -> str:
        cmd = ["nmap"]

        if scan_type == "syn":
            cmd.append("-sS")
        elif scan_type == "tcp":
            cmd.append("-sT")
        elif scan_type == "udp":
            cmd.append("-sU")
        elif scan_type == "ping":
            cmd.append("-sn")

        if ports:
            cmd.extend(["-p", ports])

        if os_detect:
            cmd.append("-O")

        if version_detect:
            cmd.append("-sV")

        if scripts:
            cmd.extend(["--script", scripts])

        cmd.append(shlex.quote(target))

        return " ".join(cmd)

    def parse_output(self, output: str) -> Dict[str, Any]:
        """Parse nmap output into structured data."""
        result = {
            "hosts": [],
            "open_ports": [],
            "services": [],
        }

        # Parse open ports
        port_pattern = r'(\d+)/(\w+)\s+open\s+(\S+)'
        for match in re.finditer(port_pattern, output):
            port, protocol, service = match.groups()
            result["open_ports"].append({
                "port": int(port),
                "protocol": protocol,
                "service": service,
            })

        # Parse host status
        if "Host is up" in output:
            result["host_up"] = True

        return result


class NiktoTool(SecurityTool):
    """Web server vulnerability scanner."""

    def __init__(self):
        super().__init__(
            name="nikto",
            description="Web server scanner for vulnerabilities and misconfigurations",
            category=ToolCategory.SCANNING,
            parameters={
                "target": {"type": "string", "required": True, "description": "Target URL or IP"},
                "port": {"type": "integer", "required": False, "description": "Target port"},
                "ssl": {"type": "boolean", "required": False, "description": "Use SSL/HTTPS"},
            },
            examples=[
                "nikto -h http://192.168.1.100",
                "nikto -h 10.0.0.1 -p 8080",
            ]
        )

    def build_command(self, target: str, port: int = None, ssl: bool = False, **kwargs) -> str:
        cmd = ["nikto", "-h", shlex.quote(target)]

        if port:
            cmd.extend(["-p", str(port)])

        if ssl:
            cmd.append("-ssl")

        return " ".join(cmd)


class GobusterTool(SecurityTool):
    """Directory and file brute-forcer."""

    def __init__(self):
        super().__init__(
            name="gobuster",
            description="Directory/file brute-forcer for web servers",
            category=ToolCategory.ENUMERATION,
            parameters={
                "target": {"type": "string", "required": True, "description": "Target URL"},
                "wordlist": {"type": "string", "required": True, "description": "Path to wordlist"},
                "extensions": {"type": "string", "required": False, "description": "File extensions to search (e.g., 'php,html')"},
                "threads": {"type": "integer", "required": False, "description": "Number of threads"},
            },
            examples=[
                "gobuster dir -u http://target.com -w /usr/share/wordlists/dirb/common.txt",
            ]
        )

    def build_command(self, target: str, wordlist: str, extensions: str = None,
                      threads: int = None, **kwargs) -> str:
        cmd = ["gobuster", "dir", "-u", shlex.quote(target), "-w", shlex.quote(wordlist)]

        if extensions:
            cmd.extend(["-x", extensions])

        if threads:
            cmd.extend(["-t", str(threads)])

        return " ".join(cmd)


class NetcatTool(SecurityTool):
    """Network utility for connections and listeners."""

    def __init__(self):
        super().__init__(
            name="nc",
            description="Netcat - network utility for TCP/UDP connections",
            category=ToolCategory.UTILITY,
            parameters={
                "mode": {"type": "string", "required": True, "enum": ["listen", "connect"], "description": "Listen or connect mode"},
                "host": {"type": "string", "required": False, "description": "Target host (for connect mode)"},
                "port": {"type": "integer", "required": True, "description": "Port number"},
            },
            examples=[
                "nc -lvnp 4444",  # Listen
                "nc 192.168.1.100 4444",  # Connect
            ]
        )

    def build_command(self, mode: str, port: int, host: str = None, **kwargs) -> str:
        if mode == "listen":
            return f"nc -lvnp {port}"
        else:
            return f"nc {shlex.quote(host)} {port}"


class CurlTool(SecurityTool):
    """HTTP client for web requests."""

    def __init__(self):
        super().__init__(
            name="curl",
            description="HTTP client for making web requests",
            category=ToolCategory.UTILITY,
            parameters={
                "url": {"type": "string", "required": True, "description": "Target URL"},
                "method": {"type": "string", "required": False, "enum": ["GET", "POST", "PUT", "DELETE"], "description": "HTTP method"},
                "data": {"type": "string", "required": False, "description": "POST data"},
                "headers": {"type": "array", "required": False, "description": "HTTP headers"},
            },
            examples=[
                "curl http://target.com",
                "curl -X POST -d 'user=admin' http://target.com/login",
            ]
        )

    def build_command(self, url: str, method: str = "GET", data: str = None,
                      headers: List[str] = None, **kwargs) -> str:
        cmd = ["curl", "-s"]

        if method != "GET":
            cmd.extend(["-X", method])

        if data:
            cmd.extend(["-d", shlex.quote(data)])

        if headers:
            for header in headers:
                cmd.extend(["-H", shlex.quote(header)])

        cmd.append(shlex.quote(url))

        return " ".join(cmd)


class SearchsploitTool(SecurityTool):
    """Exploit database search."""

    def __init__(self):
        super().__init__(
            name="searchsploit",
            description="Search exploit-db for known exploits",
            category=ToolCategory.RECON,
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query (service/version)"},
            },
            examples=[
                "searchsploit apache 2.4",
                "searchsploit vsftpd 2.3.4",
            ]
        )

    def build_command(self, query: str, **kwargs) -> str:
        return f"searchsploit {shlex.quote(query)}"


# =============================================================================
# Tool Registry
# =============================================================================

SECURITY_TOOLS = {
    "nmap": NmapTool(),
    "nikto": NiktoTool(),
    "gobuster": GobusterTool(),
    "nc": NetcatTool(),
    "curl": CurlTool(),
    "searchsploit": SearchsploitTool(),
}


def get_tool(name: str) -> Optional[SecurityTool]:
    """Get a tool by name."""
    return SECURITY_TOOLS.get(name.lower())


def get_all_tools() -> Dict[str, SecurityTool]:
    """Get all available tools."""
    return SECURITY_TOOLS


def format_tools_for_prompt() -> str:
    """Format all tools for inclusion in agent prompt."""
    lines = ["# Available Security Tools\n"]

    for name, tool in SECURITY_TOOLS.items():
        lines.append(f"## {name}")
        lines.append(f"Category: {tool.category.value}")
        lines.append(f"Description: {tool.description}")
        lines.append("Parameters:")
        for param, spec in tool.parameters.items():
            req = "(required)" if spec.get("required") else "(optional)"
            lines.append(f"  - {param} {req}: {spec.get('description', '')}")
        lines.append(f"Examples: {', '.join(tool.examples[:2])}")
        lines.append("")

    return "\n".join(lines)
