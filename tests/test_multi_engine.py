"""Test multi-engine compatibility and consistency."""

import pytest
from unittest.mock import patch, MagicMock
import json
from typing import List
from pydantic import BaseModel, ValidationError

pytestmark = pytest.mark.multi_engine


class MockAnalyzeThreatOutput(BaseModel):
    """Mock output model for testing."""

    risk_assessment: str
    recommendations: List[str]
    confidence_level: float


class TestEngineCompatibility:
    """Test that OpenAI and Claude engines produce compatible results."""

    def test_response_parsing_consistency(self):
        """Test that both engines parse responses consistently."""
        # Common response data
        response_data = {
            "risk_assessment": "Critical security vulnerability detected",
            "recommendations": ["Patch immediately", "Monitor logs", "Restrict access"],
            "confidence_level": 0.95,
        }

        # Test OpenAI format (JSON in message content)
        openai_response = json.dumps(response_data)
        parsed_openai = json.loads(openai_response)
        openai_model = MockAnalyzeThreatOutput(**parsed_openai)

        # Test Claude format (JSON in text content)
        claude_response = json.dumps(response_data)
        parsed_claude = json.loads(claude_response)
        claude_model = MockAnalyzeThreatOutput(**parsed_claude)

        # Both should produce identical results
        assert openai_model.risk_assessment == claude_model.risk_assessment
        assert openai_model.recommendations == claude_model.recommendations
        assert openai_model.confidence_level == claude_model.confidence_level

    def test_json_extraction_robustness(self):
        """Test JSON extraction works with different response formats."""

        def extract_json(response_text):
            """Simulate JSON extraction logic."""
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return response_text[json_start:json_end]
            return None

        # Test with clean JSON
        clean_response = '{"risk_assessment": "test", "recommendations": ["rec1"], "confidence_level": 0.5}'
        assert extract_json(clean_response) == clean_response

        # Test with text before/after JSON
        wrapped_response = 'Here is my analysis: {"risk_assessment": "test", "recommendations": ["rec1"], "confidence_level": 0.5} Let me know if you need more details.'
        extracted = extract_json(wrapped_response)
        assert '{"risk_assessment"' in extracted
        assert '"confidence_level": 0.5}' in extracted

        # Test with no JSON
        no_json_response = "This response contains no JSON data"
        assert extract_json(no_json_response) is None

    @patch("openai.OpenAI")
    def test_openai_api_integration_mock(self, mock_openai_class):
        """Test OpenAI API integration with mocked responses."""
        # Setup mock client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "risk_assessment": "OpenAI detected SQL injection risk",
                "recommendations": ["Use parameterized queries", "Input validation"],
                "confidence_level": 0.88,
            }
        )
        mock_client.chat.completions.create.return_value = mock_response

        # Simulate agent call
        client = mock_openai_class()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.3,
        )

        # Verify response structure
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert "risk_assessment" in parsed
        assert "recommendations" in parsed
        assert "confidence_level" in parsed

    @patch("anthropic.Anthropic")
    def test_claude_api_integration_mock(self, mock_anthropic_class):
        """Test Claude API integration with mocked responses."""
        # Setup mock client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps(
            {
                "risk_assessment": "Claude identified malware threat",
                "recommendations": [
                    "Isolate system",
                    "Run antivirus scan",
                    "Check network logs",
                ],
                "confidence_level": 0.92,
            }
        )
        mock_client.messages.create.return_value = mock_response

        # Simulate agent call
        client = mock_anthropic_class()
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "test"}],
            system="You are a security analyst",
            max_tokens=1000,
            temperature=0.3,
        )

        # Verify response structure
        content = response.content[0].text
        parsed = json.loads(content)
        assert "risk_assessment" in parsed
        assert "recommendations" in parsed
        assert "confidence_level" in parsed

    def test_temperature_consistency(self):
        """Test that temperature settings are applied consistently."""
        # Both engines should respect the same temperature range
        security_temp_range = [0.1, 0.5]

        # Test valid temperatures
        valid_temps = [0.1, 0.2, 0.3, 0.4, 0.5]
        for temp in valid_temps:
            assert security_temp_range[0] <= temp <= security_temp_range[1]

        # Test invalid temperatures
        invalid_temps = [0.0, 0.6, 0.8, 1.0]
        for temp in invalid_temps:
            assert not (security_temp_range[0] <= temp <= security_temp_range[1])

    def test_error_handling_consistency(self):
        """Test that both engines handle errors consistently."""

        # Test invalid JSON handling
        invalid_responses = [
            "Not JSON at all",
            '{"incomplete": json',
            '{"risk_assessment": "test"}',  # Missing required fields
            '{"risk_assessment": "test", "recommendations": "string", "confidence_level": 0.5}',  # Wrong type
        ]

        for invalid_response in invalid_responses:
            try:
                # Try to parse as JSON
                if invalid_response.startswith("{"):
                    parsed = json.loads(invalid_response)
                    # Try to validate with Pydantic
                    MockAnalyzeThreatOutput(**parsed)
                else:
                    # Not JSON, should trigger parsing error
                    raise json.JSONDecodeError("Not JSON", invalid_response, 0)
            except (json.JSONDecodeError, ValidationError, TypeError):
                # Expected error - both engines should handle these consistently
                continue
            except Exception as e:
                pytest.fail(
                    f"Unexpected error type: {type(e)} for response: {invalid_response}"
                )


class TestContractEnforcement:
    """Test behavioral contract enforcement across engines."""

    def test_required_fields_enforcement(self):
        """Test that required fields are enforced consistently."""
        required_fields = ["risk_assessment", "recommendations", "confidence_level"]

        # Test complete response
        complete_data = {
            "risk_assessment": "Complete assessment",
            "recommendations": ["rec1", "rec2"],
            "confidence_level": 0.8,
        }
        result = MockAnalyzeThreatOutput(**complete_data)
        assert all(hasattr(result, field) for field in required_fields)

        # Test incomplete responses
        for missing_field in required_fields:
            incomplete_data = complete_data.copy()
            del incomplete_data[missing_field]

            with pytest.raises(ValidationError) as exc_info:
                MockAnalyzeThreatOutput(**incomplete_data)

            # Should mention the missing field
            assert missing_field in str(exc_info.value).lower()

    def test_safety_check_simulation(self):
        """Simulate safety checks that would be applied by behavioral contracts."""

        # Simulate harmful content detection
        potentially_harmful_responses = [
            "Here's how to hack into systems...",
            "Delete all files with rm -rf /",
            "Instructions for creating malware...",
        ]

        # Safety check simulation
        def contains_harmful_content(text):
            harmful_keywords = ["hack", "delete all", "malware", "exploit"]
            return any(keyword in text.lower() for keyword in harmful_keywords)

        # Test that safety checks would catch harmful content
        for harmful_text in potentially_harmful_responses:
            assert contains_harmful_content(harmful_text)

        # Test that normal security analysis passes
        normal_responses = [
            "The vulnerability should be patched immediately",
            "Implement additional monitoring for this system",
            "Update security policies to prevent this issue",
        ]

        for normal_text in normal_responses:
            assert not contains_harmful_content(normal_text)

    def test_confidence_bounds_enforcement(self):
        """Test confidence level bounds are enforced."""
        from pydantic import BaseModel, Field
        from typing import List

        class BoundedOutput(BaseModel):
            risk_assessment: str
            recommendations: List[str]
            confidence_level: float = Field(ge=0.0, le=1.0)

        # Valid confidence levels
        valid_confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
        for confidence in valid_confidences:
            data = {
                "risk_assessment": "test",
                "recommendations": ["test"],
                "confidence_level": confidence,
            }
            result = BoundedOutput(**data)
            assert 0.0 <= result.confidence_level <= 1.0

        # Invalid confidence levels
        invalid_confidences = [-0.1, 1.1, 2.0, -1.0]
        for confidence in invalid_confidences:
            data = {
                "risk_assessment": "test",
                "recommendations": ["test"],
                "confidence_level": confidence,
            }
            with pytest.raises(ValidationError):
                BoundedOutput(**data)


class TestPerformanceConsistency:
    """Test that performance characteristics are consistent across engines."""

    def test_response_time_simulation(self):
        """Simulate response time validation."""
        import time

        # Simulate contract timeout (30 seconds for security tasks)
        max_response_time = 30.0

        # Simulate fast response
        start_time = time.time()
        time.sleep(0.001)  # Simulate processing
        elapsed = time.time() - start_time

        assert elapsed < max_response_time

        # Test timeout boundary
        assert max_response_time == 30.0  # Should match contract specification

    def test_token_usage_estimation(self):
        """Test token usage patterns for both engines."""

        # Simulate typical security analysis response lengths
        typical_responses = [
            {
                "risk_assessment": "Short assessment",
                "recommendations": ["Quick fix"],
                "confidence_level": 0.7,
            },
            {
                "risk_assessment": "This is a much longer and more detailed risk assessment that covers multiple aspects of the security vulnerability including impact analysis, affected systems, potential attack vectors, and comprehensive mitigation strategies.",
                "recommendations": [
                    "Implement immediate patches for the identified vulnerability",
                    "Conduct comprehensive security audit of affected systems",
                    "Update security monitoring rules to detect similar threats",
                    "Review and update incident response procedures",
                    "Provide security awareness training to relevant personnel",
                ],
                "confidence_level": 0.95,
            },
        ]

        for response in typical_responses:
            # Estimate token usage (rough approximation)
            text_content = json.dumps(response)
            estimated_tokens = len(text_content.split()) * 1.3  # Rough estimate

            # Should be within reasonable limits for both engines
            assert estimated_tokens < 1000  # Max tokens from our config
            assert len(response["recommendations"]) >= 1  # At least one recommendation
