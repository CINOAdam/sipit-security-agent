"""
Metasploitable2 exploitation goals.

Uses port forwards through 192.168.0.41:
- FTP: 8021 (vsftpd 2.3.4 with backdoor)
- SSH: 8022 (OpenSSH 4.7p1)
- Telnet: 8023
- HTTP: 8080
- SMB: 8445
"""

from security_agent.goals import Goal, GoalType, Target

# Metasploitable2 target with port forwards
METASPLOITABLE = Target(
    ip="192.168.0.41",
    hostname="metasploitable2",
    known_services=[
        {"port": 8021, "service": "ftp", "version": "vsftpd 2.3.4"},
        {"port": 8022, "service": "ssh", "version": "OpenSSH 4.7p1"},
        {"port": 8023, "service": "telnet"},
        {"port": 8080, "service": "http", "version": "Apache"},
        {"port": 8445, "service": "smb"},
    ],
    notes="Metasploitable2 via port forwards through OPNsense"
)


METASPLOIT_GOALS = [
    # -------------------------------------------------------------------------
    # EASY: Enumeration
    # -------------------------------------------------------------------------
    Goal(
        id="msf_full_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate all Metasploitable2 services through port forwards",
        target=METASPLOITABLE,
        success_criteria="Identify all open services and their versions",
        hints=[
            "Ports: 8021 (FTP), 8022 (SSH), 8023 (Telnet), 8080 (HTTP), 8445 (SMB)",
            "Use nmap with version detection",
        ],
        max_steps=6,
    ),

    Goal(
        id="msf_ftp_enum",
        type=GoalType.ENUMERATE,
        description="Enumerate the FTP service on Metasploitable2",
        target=METASPLOITABLE,
        success_criteria="Identify FTP version and check for anonymous access",
        hints=[
            "FTP on port 8021",
            "Check for vsftpd version",
            "Try anonymous login",
        ],
        max_steps=5,
    ),

    # -------------------------------------------------------------------------
    # HARD: Exploitation
    # -------------------------------------------------------------------------
    Goal(
        id="msf_vsftpd_backdoor",
        type=GoalType.GET_SHELL,
        description="Exploit the vsftpd 2.3.4 backdoor to get a shell",
        target=METASPLOITABLE,
        success_criteria="Trigger the vsftpd backdoor and connect to shell on port 6200",
        hints=[
            "vsftpd 2.3.4 has a backdoor triggered by ':)' in username",
            "Send USER with smiley face: USER backdoor:)",
            "Backdoor opens shell on port 6200 (need to check if forwarded)",
            "Connect with nc to the backdoor port",
        ],
        max_steps=10,
    ),

    Goal(
        id="msf_ssh_bruteforce",
        type=GoalType.CREDENTIAL,
        description="Find valid SSH credentials for Metasploitable2",
        target=METASPLOITABLE,
        success_criteria="Discover working SSH username/password",
        hints=[
            "SSH on port 8022",
            "Common credentials: msfadmin/msfadmin",
            "Try default credentials first",
        ],
        max_steps=8,
    ),

    Goal(
        id="msf_http_exploit",
        type=GoalType.ENUMERATE,
        description="Find and exploit vulnerabilities in Metasploitable2 web server",
        target=METASPLOITABLE,
        success_criteria="Identify web vulnerabilities and potential exploits",
        hints=[
            "HTTP on port 8080",
            "Check for vulnerable web apps",
            "Look for phpMyAdmin, DVWA, etc.",
        ],
        max_steps=10,
    ),
]


if __name__ == "__main__":
    print("Metasploitable2 Goals:")
    for goal in METASPLOIT_GOALS:
        print(f"  - {goal.id}: {goal.description}")
