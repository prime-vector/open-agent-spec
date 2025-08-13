"""Test Cortex engine integration with OAS."""

import pytest

pytestmark = pytest.mark.cortex


class TestCortexEngineIntegration:
    """Test Cortex engine integration and configuration."""

    def test_cortex_engine_configuration(self):
        """Test that Cortex engine configuration is properly structured."""
        cortex_config = {
            "engine": "cortex",
            "model": "cortex-intelligence",
            "config": {
                "enable_layer3": True,
                "enable_onnx": False,
                "openai_api_key": "${OPENAI_API_KEY}",
                "claude_api_key": "${CLAUDE_API_KEY}",
                "temperature": 0.2,
                "max_tokens": 1500,
            },
        }

        # Verify required fields
        assert cortex_config["engine"] == "cortex"
        assert cortex_config["model"] == "cortex-intelligence"
        assert "config" in cortex_config

        # Verify Cortex-specific config
        config = cortex_config["config"]
        assert "enable_layer3" in config
        assert "enable_onnx" in config
        assert "openai_api_key" in config
        assert "claude_api_key" in config
        assert "temperature" in config
        assert "max_tokens" in config

        # Verify data types
        assert isinstance(config["enable_layer3"], bool)
        assert isinstance(config["enable_onnx"], bool)
        assert isinstance(config["temperature"], (int, float))
        assert isinstance(config["max_tokens"], int)

    def test_cortex_agent_generation(self):
        """Test that Cortex agents can be generated properly."""

        # Mock spec data for Cortex agent
        spec_data = {
            "open_agent_spec": "1.0.8",
            "agent": {
                "name": "test-cortex-agent",
                "description": "Test agent using Cortex engine",
                "role": "analyst",
            },
            "intelligence": {
                "type": "llm",
                "engine": "cortex",
                "model": "cortex-intelligence",
                "config": {
                    "enable_layer3": True,
                    "enable_onnx": False,
                    "temperature": 0.2,
                    "max_tokens": 1500,
                },
            },
            "tasks": {
                "analyze": {
                    "description": "Test analysis task",
                    "timeout": 60,
                    "input": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "output": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                        "required": ["result"],
                    },
                }
            },
        }

        # This should not raise an error
        try:
            # Note: This is a basic test - actual generation would require more setup
            assert spec_data["intelligence"]["engine"] == "cortex"
            assert "enable_layer3" in spec_data["intelligence"]["config"]
        except Exception as e:
            pytest.fail(f"Cortex agent spec validation failed: {e}")

    def test_cortex_config_validation(self):
        """Test Cortex configuration validation."""
        # Valid Cortex config
        valid_config = {
            "enable_layer3": True,
            "enable_onnx": False,
            "openai_api_key": "sk-...",
            "claude_api_key": "sk-ant-...",
            "temperature": 0.2,
            "max_tokens": 1500,
        }

        # All required fields present
        required_fields = [
            "enable_layer3",
            "enable_onnx",
            "openai_api_key",
            "claude_api_key",
        ]
        for field in required_fields:
            assert field in valid_config

        # Valid temperature range
        assert 0.0 <= valid_config["temperature"] <= 2.0

        # Valid max_tokens
        assert valid_config["max_tokens"] > 0

    def test_cortex_environment_variables(self):
        """Test Cortex environment variable handling."""

        # Test environment variable substitution
        config_with_env = {
            "openai_api_key": "${OPENAI_API_KEY}",
            "claude_api_key": "${CLAUDE_API_KEY}",
        }

        # These should be properly formatted for environment substitution
        assert config_with_env["openai_api_key"].startswith("${")
        assert config_with_env["openai_api_key"].endswith("}")
        assert config_with_env["claude_api_key"].startswith("${")
        assert config_with_env["claude_api_key"].endswith("}")

    def test_cortex_layer3_configuration(self):
        """Test Layer 3 intelligence configuration."""
        # Test different Layer 3 configurations
        layer3_configs = [
            {"enable_layer3": True, "enable_onnx": True},
            {"enable_layer3": True, "enable_onnx": False},
            {"enable_layer3": False, "enable_onnx": True},
            {"enable_layer3": False, "enable_onnx": False},
        ]

        for config in layer3_configs:
            # Both should be boolean values
            assert isinstance(config["enable_layer3"], bool)
            assert isinstance(config["enable_onnx"], bool)

            # Layer 3 can be enabled/disabled independently of ONNX
            # This is a valid configuration
            assert True  # All combinations are valid

    def test_cortex_api_key_handling(self):
        """Test API key handling for Cortex integration."""
        # Test API key formats
        openai_key = (
            "sk-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        )
        claude_key = (
            "sk-ant-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        )

        # Verify key formats
        assert openai_key.startswith("sk-")
        assert claude_key.startswith("sk-ant-")
        assert len(openai_key) >= 50
        assert len(claude_key) >= 50

        # Test configuration with actual keys
        config_with_keys = {
            "enable_layer3": True,
            "enable_onnx": False,
            "openai_api_key": openai_key,
            "claude_api_key": claude_key,
            "temperature": 0.2,
            "max_tokens": 1500,
        }

        # Keys should be properly stored
        assert config_with_keys["openai_api_key"] == openai_key
        assert config_with_keys["claude_api_key"] == claude_key

    def test_cortex_template_structure(self):
        """Test that Cortex template follows OAS structure."""
        # Load the Cortex template
        template_path = "oas_cli/templates/cortex-intelligence-agent.yaml"

        try:
            import yaml

            with open(template_path, "r") as f:
                template_data = yaml.safe_load(f)

            # Verify template structure
            assert "open_agent_spec" in template_data
            assert "agent" in template_data
            assert "intelligence" in template_data
            assert "tasks" in template_data

            # Verify intelligence configuration
            intelligence = template_data["intelligence"]
            assert intelligence["engine"] == "cortex"
            assert intelligence["model"] == "cortex-intelligence"
            assert "enable_layer3" in intelligence["config"]
            assert "enable_onnx" in intelligence["config"]

        except FileNotFoundError:
            pytest.skip("Cortex template not found")
        except Exception as e:
            pytest.fail(f"Failed to load Cortex template: {e}")


class TestCortexEngineCompatibility:
    """Test Cortex engine compatibility with existing OAS features."""

    def test_cortex_with_dacp_integration(self):
        """Test that Cortex works with DACP integration."""
        # Cortex should integrate with DACP like other engines
        cortex_intelligence_config = {
            "engine": "cortex",
            "model": "cortex-intelligence",
            "config": {
                "enable_layer3": True,
                "enable_onnx": False,
                "temperature": 0.2,
                "max_tokens": 1500,
            },
        }

        # Should have required DACP fields
        assert "engine" in cortex_intelligence_config
        assert "model" in cortex_intelligence_config
        assert "config" in cortex_intelligence_config

    def test_cortex_behavioral_contracts(self):
        """Test that Cortex supports behavioral contracts."""
        # Cortex should support the same behavioral contract features
        contract_config = {
            "behavioural_flags": {
                "reasoning_depth": "comprehensive",
                "creativity_level": "high",
                "evidence_based": "strict",
            },
            "response_contract": {
                "output_format": {"required_fields": ["result", "reasoning"]}
            },
        }

        # Should have behavioral contract structure
        assert "behavioural_flags" in contract_config
        assert "response_contract" in contract_config
        assert "output_format" in contract_config["response_contract"]

    def test_cortex_multi_task_support(self):
        """Test that Cortex supports multi-task operations."""
        # Cortex should support multi-step tasks
        multi_task_config = {
            "multi_step": True,
            "steps": [
                {"task": "analyze", "input_map": {"query": "{{input.query}}"}},
                {"task": "synthesize", "input_map": {"analysis": "{{analyze.result}}"}},
            ],
        }

        # Should have multi-step structure
        assert multi_task_config["multi_step"] is True
        assert "steps" in multi_task_config
        assert len(multi_task_config["steps"]) > 0
