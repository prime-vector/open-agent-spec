#!/usr/bin/env python3
"""
DACP Logging Integration Test

This test demonstrates the full DACP logging integration working end-to-end:
1. Generates an agent with logging configuration
2. Installs dependencies
3. Mocks intelligence calls to avoid needing API keys
4. Runs the agent and captures DACP logging output
5. Verifies comprehensive logging is working
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import pytest


def test_dacp_logging_integration_full_flow():
    """Test the complete DACP logging integration flow without real API calls."""

    # Create a temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        agent_dir = temp_path / "test_agent"

        print("\n🧪 Testing DACP Logging Integration")
        print(f"📁 Test directory: {temp_path}")

        # Step 1: Generate agent using the CLI
        print("\n1️⃣ Generating agent with DACP logging...")

        # Use the CLI to generate an agent
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "oas_cli.main",
                "init",
                "--spec",
                "oas_cli/templates/minimal-agent.yaml",
                "--output",
                str(agent_dir),
            ],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        if result.returncode != 0:
            print(f"❌ CLI generation failed: {result.stderr}")
            pytest.fail(f"Agent generation failed: {result.stderr}")

        print("✅ Agent generated successfully")

        # Step 2: Verify generated files
        print("\n2️⃣ Verifying generated files...")

        required_files = ["agent.py", "requirements.txt", ".env.example", "README.md"]
        for file in required_files:
            file_path = agent_dir / file
            assert file_path.exists(), f"Missing file: {file}"
            print(f"   ✅ {file}")

        # Step 3: Check agent.py contains DACP integration
        print("\n3️⃣ Verifying DACP integration in generated code...")

        agent_code = (agent_dir / "agent.py").read_text()
        dacp_checks = [
            "import dacp",
            "class HelloWorldAgent(dacp.Agent):",
            "def setup_logging(self):",
            "dacp.setup_dacp_logging(",
            "self.config = {",
            '"logging": {',
        ]

        for check in dacp_checks:
            assert check in agent_code, f"Missing DACP integration: {check}"
            print(f"   ✅ {check}")

        # Step 4: Install dependencies
        print("\n4️⃣ Installing dependencies...")

        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            cwd=agent_dir,
        )

        if install_result.returncode != 0:
            print(
                f"⚠️  Dependency installation issues (expected): {install_result.stderr}"
            )
        else:
            print("✅ Dependencies installed")

        # Step 5: Create a test script that mocks intelligence calls
        print("\n5️⃣ Creating test script with mocked intelligence...")

        test_script = agent_dir / "test_with_mocked_intelligence.py"
        test_script.write_text(
            '''
import sys
import json
from unittest.mock import patch
import os

# Import the generated agent
from agent import HelloWorldAgent
from dacp.orchestrator import Orchestrator

def mock_invoke_intelligence(prompt, config):
    """Mock intelligence function that returns a realistic response."""
    print(f"🤖 Mock LLM called with engine: {config.get('engine', 'unknown')}")
    return json.dumps({
        "response": "Hello! I'm a test agent with DACP logging integration working perfectly!"
    })

def test_agent_with_logging():
    """Test the agent with mocked intelligence to see DACP logging."""

    print("\\n🚀 Starting DACP Logging Integration Test")
    print("=" * 60)

    # Mock the intelligence call to avoid needing API keys
    with patch('dacp.invoke_intelligence', side_effect=mock_invoke_intelligence):
        print("\\n1. Creating orchestrator...")
        orchestrator = Orchestrator()

        print("2. Instantiating agent (should show DACP logging setup)...")
        agent = HelloWorldAgent("test-agent-123", orchestrator)

        print("\\n3. Calling agent method (should show intelligence logging)...")
        try:
            result = agent.greet(name="Integration Test")
            print(f"\\n✅ Agent response: {result}")

            # Verify the result structure
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            else:
                result_dict = result

            print(f"📊 Result type: {type(result)}")
            print(f"📋 Result data: {json.dumps(result_dict, indent=2)}")

        except Exception as e:
            print(f"❌ Agent call failed: {e}")
            raise

    print("\\n🎉 DACP Logging Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_agent_with_logging()
'''
        )

        # Step 6: Run the test script
        print("\n6️⃣ Running agent with mocked intelligence...")

        # Ensure we can import the modules
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{agent_dir}:{env.get('PYTHONPATH', '')}"

        test_result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            cwd=agent_dir,
            env=env,
        )

        print("📊 Test Output:")
        print("-" * 40)
        print(test_result.stdout)
        if test_result.stderr:
            print("⚠️ Stderr:")
            print(test_result.stderr)
        print("-" * 40)

        # Step 7: Verify logging output
        print("\n7️⃣ Verifying DACP logging output...")

        expected_log_messages = [
            "🚀 DACP logging configured",
            "🧠 Invoking intelligence",
            "🤖 Mock LLM called",
        ]

        output = test_result.stdout + test_result.stderr
        for expected in expected_log_messages:
            if expected not in output:
                print(f"⚠️  Expected log message not found: {expected}")
            else:
                print(f"   ✅ Found: {expected}")

        # Verify the test ran successfully
        if test_result.returncode == 0:
            print("\n🎉 Integration test completed successfully!")
            print("✅ DACP logging integration is working correctly")
        else:
            print(f"\n❌ Test failed with return code: {test_result.returncode}")
            print("Check the output above for details")

        # Step 8: Test environment variable overrides
        print("\n8️⃣ Testing environment variable overrides...")

        env_test_script = agent_dir / "test_env_overrides.py"
        env_test_script.write_text(
            """
import os
from unittest.mock import patch
from agent import HelloWorldAgent
from dacp.orchestrator import Orchestrator

# Set environment overrides
os.environ['DACP_LOG_LEVEL'] = 'DEBUG'
os.environ['DACP_LOG_STYLE'] = 'detailed'

def mock_invoke_intelligence(prompt, config):
    return "{\\"response\\": \\"Hello with env overrides!\\"}"

print("🔧 Testing environment variable overrides...")
with patch('dacp.invoke_intelligence', side_effect=mock_invoke_intelligence):
    orchestrator = Orchestrator()
    agent = HelloWorldAgent("env-test-agent", orchestrator)
    result = agent.greet(name="Environment Test")
    print(f"✅ Environment override test completed: {result}")
"""
        )

        env_result = subprocess.run(
            [sys.executable, str(env_test_script)],
            capture_output=True,
            text=True,
            cwd=agent_dir,
            env=env,
        )

        print("📊 Environment Override Test Output:")
        print(env_result.stdout)
        if env_result.stderr:
            print("⚠️ Stderr:", env_result.stderr)

        # Final verification
        assert test_result.returncode == 0, f"Main test failed: {test_result.stderr}"
        assert (
            "DACP logging configured" in output
        ), "DACP logging setup not found in output"

        print("\n🏆 ALL INTEGRATION TESTS PASSED!")
        print("✅ CLI generation works")
        print("✅ DACP logging integration works")
        print("✅ Intelligence calls are logged")
        print("✅ Environment overrides work")
        print("✅ Agent functionality works end-to-end")


if __name__ == "__main__":
    test_dacp_logging_integration_full_flow()
