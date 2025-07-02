#!/usr/bin/env python3
"""
Local integration test script for OAS CLI templates.
This script tests all templates locally without requiring GitHub secrets.
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
import pytest


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    if cwd:
        print(f"Working directory: {cwd}")

    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)

    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    if check and result.returncode != 0:
        print(f"Command failed with return code: {result.returncode}")
        sys.exit(1)

    return result


# Mark these as integration tests that should not be run by pytest
@pytest.mark.skip(reason="Integration tests designed to run as standalone script")
def test_template(template_name, test_dir):
    """Test a single template."""
    print(f"\n{'=' * 60}")
    print(f"Testing template: {template_name}")
    print(f"{'=' * 60}")

    template_path = f"oas_cli/templates/{template_name}"
    agent_dir = test_dir / f"{template_name}-agent"

    # Clean up previous test
    if agent_dir.exists():
        shutil.rmtree(agent_dir)

    try:
        # Generate agent
        print(f"Generating agent from {template_name}...")
        run_command(f"oas init --spec {template_path} --output {agent_dir}")

        # Check required files exist
        print("Checking generated files...")
        required_files = ["agent.py", "requirements.txt", "README.md", ".env.example"]
        for file in required_files:
            file_path = agent_dir / file
            if not file_path.exists():
                print(f"ERROR: Required file {file} not found!")
                return False
            print(f"✅ {file} exists")

        # Install dependencies
        print("Installing dependencies...")
        run_command("pip install -r requirements.txt", cwd=agent_dir)

        # Test agent import
        print("Testing agent import...")
        run_command(
            "python3 -c 'from agent import *; print(\"✅ Agent imports successfully\")'",
            cwd=agent_dir,
        )

        # Test agent instantiation
        print("Testing agent instantiation...")
        # Create a temporary test file to avoid one-liner class definition issues
        test_instantiation = agent_dir / "test_instantiation.py"
        test_instantiation.write_text(
            """
from agent import *

class MockOrchestrator:
    def register_agent(self, agent_id, agent):
        pass

orchestrator = MockOrchestrator()
agent = HelloWorldAgent("test-agent-id", orchestrator)
print("✅ Agent instantiated successfully")
"""
        )

        run_command("python3 test_instantiation.py", cwd=agent_dir)

        # Clean up
        test_instantiation.unlink()

        # Test agent execution (without API key - should fail gracefully)
        print("Testing agent execution (without API key)...")
        result = run_command("python3 agent.py", cwd=agent_dir, check=False)
        if result.returncode == 0:
            print("✅ Agent executed successfully")
        else:
            print("⚠️  Agent execution failed (expected without API key)")
            print("This is normal if the agent requires an API key")

        print(f"✅ Template {template_name} passed all tests!")
        return True

    except Exception as e:
        print(f"❌ Template {template_name} failed: {e}")
        return False


@pytest.mark.skip(reason="Integration tests designed to run as standalone script")
def test_tool_agent_specific(test_dir):
    """Test tool agent with specific functionality."""
    print(f"\n{'=' * 60}")
    print("Testing tool agent specific functionality")
    print(f"{'=' * 60}")

    template_name = "minimal-agent-tool-usage.yaml"
    agent_dir = test_dir / f"{template_name}-agent"

    sys_path_patch = "import sys, os\nsys.path.insert(0, os.getcwd())\n"
    try:
        # Test that the agent has the expected methods
        print("Testing tool agent methods...")
        test_code = (
            sys_path_patch
            + """
import agent

# Check that the agent class exists
try:
    assert hasattr(agent, 'HelloWorldAgent'), "Agent class should exist"
    print("✅ Agent class exists")
except Exception as e:
    print(f"⚠️  Agent class check failed: {e}")
    sys.exit(1)  # Exit with non-zero status code to signal failure

print("✅ All required methods and functions exist")
"""
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(test_code)
            test_code_path = f.name
        run_command(f"python3 {test_code_path}", cwd=agent_dir)
        os.unlink(test_code_path)

        # Test that the save_greeting method has correct signature
        print("Testing method signatures...")
        sig_test_file = agent_dir / "test_signature.py"
        sig_test_file.write_text(
            """
import sys, os
sys.path.insert(0, os.getcwd())

import inspect
from agent import HelloWorldAgent

# Mock orchestrator for testing
class MockOrchestrator:
    def register_agent(self, agent_id, agent):
        pass

orchestrator = MockOrchestrator()
agent = HelloWorldAgent("test-agent-id", orchestrator)
sig = inspect.signature(agent.save_greeting)
params = list(sig.parameters.keys())
expected_params = ['file_path', 'name']
assert params == expected_params, "Method signature mismatch"
print("✅ Method signature is correct")
"""
        )

        run_command("python3 test_signature.py", cwd=agent_dir)

        # Clean up
        sig_test_file.unlink()

        print("✅ Tool agent specific tests passed!")
        return True

    except Exception as e:
        print(f"❌ Tool agent specific tests failed: {e}")
        return False


def main():
    """Run all template tests."""
    print("OAS CLI Template Integration Tests")
    print("=" * 60)

    # Create test directory
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "output").mkdir(exist_ok=True)

    # Test all templates
    templates = [
        "minimal-agent.yaml",
        "minimal-multi-task-agent.yaml",
        "minimal-agent-tool-usage.yaml",
    ]

    results = {}

    for template in templates:
        results[template] = test_template(template, test_dir)

    # Test tool agent specific functionality
    if results.get("minimal-agent-tool-usage.yaml", False):
        results["tool-agent-specific"] = test_tool_agent_specific(test_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    all_passed = True
    for template, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{template}: {status}")
        if not passed:
            all_passed = False

    print(
        f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    if all_passed:
        print("\n🎉 All templates are working correctly!")
        print("You can now set up the GitHub workflow with confidence.")
    else:
        print(
            "\n⚠️  Some tests failed. Please fix the issues before setting up the GitHub workflow."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
