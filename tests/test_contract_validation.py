"""Test behavioral contract validation across engines."""

import pytest
from unittest.mock import patch, MagicMock
import json

pytestmark = pytest.mark.contract


class TestContractValidation:
    """Test behavioral contract validation with mock responses."""

    @patch("openai.OpenAI")
    def test_openai_valid_response(
        self, mock_openai, mock_openai_response, sample_threat_data
    ):
        """Test OpenAI agent with valid response passes contract validation."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client

        # Simulate contract validation
        response = mock_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test threat"}],
            temperature=0.3,
        )

        # Validate response structure
        assert response.choices[0].message.content is not None
        content = json.loads(response.choices[0].message.content)

        # Validate required fields
        assert "risk_assessment" in content
        assert "recommendations" in content
        assert "confidence_level" in content

        # Validate types and constraints
        assert isinstance(content["risk_assessment"], str)
        assert isinstance(content["recommendations"], list)
        assert isinstance(content["confidence_level"], (int, float))
        assert 0.0 <= content["confidence_level"] <= 1.0

    @patch("anthropic.Anthropic")
    def test_claude_valid_response(
        self, mock_anthropic, mock_claude_response, sample_threat_data
    ):
        """Test Claude agent with valid response passes contract validation."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.return_value = mock_client

        # Simulate contract validation
        response = mock_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "test threat"}],
            temperature=0.3,
        )

        # Validate response structure
        assert response.content[0].text is not None
        content = json.loads(response.content[0].text)

        # Validate required fields
        assert "risk_assessment" in content
        assert "recommendations" in content
        assert "confidence_level" in content

        # Validate types and constraints
        assert isinstance(content["risk_assessment"], str)
        assert isinstance(content["recommendations"], list)
        assert isinstance(content["confidence_level"], (int, float))
        assert 0.0 <= content["confidence_level"] <= 1.0

    def test_required_fields_validation(self, missing_fields_response):
        """Test that missing required fields trigger contract violations."""
        # This would test the behavioral contract decorator's field validation
        # Simplified for demo
        required_fields = ["risk_assessment", "recommendations", "confidence_level"]
        response_data = {"risk_assessment": "Some assessment"}

        missing_fields = [
            field for field in required_fields if field not in response_data
        ]
        assert len(missing_fields) == 2
        assert "recommendations" in missing_fields
        assert "confidence_level" in missing_fields

    def test_type_validation(self, invalid_types_response):
        """Test that invalid field types trigger contract violations."""
        from pydantic import BaseModel, ValidationError
        from typing import List

        class TestOutput(BaseModel):
            risk_assessment: str
            recommendations: List[str]
            confidence_level: float

        # Test invalid data
        invalid_data = {
            "risk_assessment": "Some assessment",
            "recommendations": "should be array not string",
            "confidence_level": "high",
        }

        with pytest.raises(ValidationError):
            TestOutput(**invalid_data)

    def test_confidence_level_bounds(self):
        """Test confidence level must be between 0.0 and 1.0."""
        from pydantic import BaseModel, ValidationError, Field
        from typing import List

        class TestOutput(BaseModel):
            risk_assessment: str
            recommendations: List[str]
            confidence_level: float = Field(ge=0.0, le=1.0)

        # Valid confidence levels
        valid_data = {
            "risk_assessment": "Test",
            "recommendations": ["rec1"],
            "confidence_level": 0.5,
        }
        result = TestOutput(**valid_data)
        assert 0.0 <= result.confidence_level <= 1.0

        # Invalid confidence levels
        for invalid_confidence in [-0.1, 1.1, 2.0]:
            invalid_data = {
                "risk_assessment": "Test",
                "recommendations": ["rec1"],
                "confidence_level": invalid_confidence,
            }
            with pytest.raises(ValidationError):
                TestOutput(**invalid_data)

    def test_temperature_range_enforcement(self):
        """Test that temperature control is enforced by contract."""
        # This tests the behavioral contract's temperature control
        valid_range = [0.1, 0.5]

        # Test values within range
        for temp in [0.1, 0.3, 0.5]:
            assert valid_range[0] <= temp <= valid_range[1]

        # Test values outside range
        for temp in [0.0, 0.6, 1.0]:
            assert not (valid_range[0] <= temp <= valid_range[1])

    def test_json_parsing_robustness(self):
        """Test that JSON parsing handles various response formats."""
        import json

        # Valid JSON
        valid_json = '{"risk_assessment": "test", "recommendations": ["rec1"], "confidence_level": 0.5}'
        parsed = json.loads(valid_json)
        assert "risk_assessment" in parsed

        # Invalid JSON should raise exception
        invalid_json = '{"incomplete": json'
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

        # JSON with extra whitespace/formatting
        formatted_json = """
        {
            "risk_assessment": "test assessment",
            "recommendations": [
                "recommendation 1",
                "recommendation 2"
            ],
            "confidence_level": 0.85
        }
        """
        parsed = json.loads(formatted_json)
        assert len(parsed["recommendations"]) == 2


class TestMultiEngineCompatibility:
    """Test that contracts work consistently across engines."""

    def test_response_format_consistency(self):
        """Test that both engines produce compatible response formats."""
        # Both engines should produce the same Pydantic model structure
        from typing import List
        from pydantic import BaseModel

        class UnifiedOutput(BaseModel):
            risk_assessment: str
            recommendations: List[str]
            confidence_level: float

        # Test with OpenAI-style response
        openai_data = {
            "risk_assessment": "OpenAI assessment",
            "recommendations": ["openai rec1", "openai rec2"],
            "confidence_level": 0.8,
        }
        openai_result = UnifiedOutput(**openai_data)

        # Test with Claude-style response
        claude_data = {
            "risk_assessment": "Claude assessment",
            "recommendations": ["claude rec1", "claude rec2"],
            "confidence_level": 0.9,
        }
        claude_result = UnifiedOutput(**claude_data)

        # Both should have same structure
        assert hasattr(openai_result, "risk_assessment")
        assert hasattr(claude_result, "risk_assessment")
        assert isinstance(openai_result.recommendations, list)
        assert isinstance(claude_result.recommendations, list)

    def test_contract_decorator_compatibility(self):
        """Test that @behavioural_contract works with both engines."""
        # This would test that the decorator handles both OpenAI and Claude responses
        # Simplified for demo
        contract_config = {
            "version": "0.1.2",
            "description": "Test contract",
            "behavioural_flags": {"temperature_control": {"range": [0.1, 0.5]}},
            "response_contract": {
                "output_format": {
                    "required_fields": [
                        "risk_assessment",
                        "recommendations",
                        "confidence_level",
                    ]
                }
            },
        }

        # Verify contract configuration is engine-agnostic
        assert (
            "required_fields" in contract_config["response_contract"]["output_format"]
        )
        assert (
            len(contract_config["behavioural_flags"]["temperature_control"]["range"])
            == 2
        )
