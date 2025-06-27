"""Pytest configuration and fixtures for framework tests."""

import pytest
from unittest.mock import MagicMock
import json


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response with valid security analysis."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "risk_assessment": "High risk SQL injection vulnerability detected in login system",
            "recommendations": [
                "Implement parameterized queries",
                "Add input validation",
                "Deploy WAF protection",
            ],
            "confidence_level": 0.85,
        }
    )
    return mock_response


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response with valid security analysis."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps(
        {
            "risk_assessment": "Critical malware infection spreading through network shares",
            "recommendations": [
                "Isolate affected systems immediately",
                "Run full antivirus scan",
                "Review network access logs",
            ],
            "confidence_level": 0.92,
        }
    )
    return mock_response


@pytest.fixture
def missing_fields_response():
    """Mock response missing required fields to test validation."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "risk_assessment": "Some assessment",
            # Missing recommendations and confidence_level
        }
    )
    return mock_response


@pytest.fixture
def invalid_types_response():
    """Mock response with invalid field types to test validation."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "risk_assessment": "Some assessment",
            "recommendations": "should be array not string",  # Wrong type
            "confidence_level": "high",  # Should be number
        }
    )
    return mock_response


@pytest.fixture
def sample_threat_data():
    """Sample threat analysis input data."""
    return {
        "threat_description": "SQL injection attempts detected in web application login forms",
        "severity": "high",
        "system_affected": "Customer portal database",
    }
