#!/usr/bin/env python3
"""Test security tools against real targets (no LLM needed)."""

import asyncio
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from security_agent.tools import SECURITY_TOOLS


async def test_tools():
    """Test each tool against real targets."""
    print("=" * 70)
    print("SECURITY TOOLS - LIVE TEST")
    print("=" * 70)
    print("Testing tool execution against your homelab targets\n")

    tests = [
        # Test 1: Curl DVWA
        {
            "name": "curl → DVWA",
            "tool": "curl",
            "params": {"url": "http://192.168.0.41:8001"},
            "expect": "login",
        },
        # Test 2: Curl Juice Shop
        {
            "name": "curl → Juice Shop",
            "tool": "curl",
            "params": {"url": "http://192.168.0.41:8002"},
            "expect": "OWASP",
        },
        # Test 3: Curl VAmPI
        {
            "name": "curl → VAmPI API",
            "tool": "curl",
            "params": {"url": "http://192.168.0.41:8008"},
            "expect": "api",
        },
        # Test 4: Nmap quick scan (just check if it runs)
        {
            "name": "nmap → DVWA (port 8001)",
            "tool": "nmap",
            "params": {"target": "192.168.0.41", "ports": "8001", "scan_type": "tcp"},
            "expect": "open",
        },
    ]

    results = []
    for test in tests:
        print(f"\n{'─' * 60}")
        print(f"Test: {test['name']}")
        print(f"{'─' * 60}")

        tool = SECURITY_TOOLS.get(test["tool"])
        if not tool:
            print(f"  ✗ Tool not found: {test['tool']}")
            continue

        try:
            result = await tool.execute(timeout=30, **test["params"])

            print(f"  Command: {result.command}")
            print(f"  Success: {result.success}")

            if result.output:
                preview = result.output[:300].replace('\n', ' ')
                print(f"  Output: {preview}...")

                # Check for expected content
                if test["expect"].lower() in result.output.lower():
                    print(f"  ✓ Found expected: '{test['expect']}'")
                    results.append((test["name"], True))
                else:
                    print(f"  ⚠ Expected '{test['expect']}' not found")
                    results.append((test["name"], False))
            else:
                print(f"  Output: (empty)")
                results.append((test["name"], False))

            if result.error:
                print(f"  Error: {result.error[:100]}")

        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append((test["name"], False))

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(test_tools())
    sys.exit(0 if success else 1)
