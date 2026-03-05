"""Tests for improved validation error messages."""

import pytest

from oas_cli.validators import validate_spec


class TestImprovedErrorMessages:
    """Test that validation errors provide helpful, actionable messages."""

    def test_missing_open_agent_spec_version(self):
        """Error message should explain what's missing and show example."""
        spec_data = {
            "agent": {"name": "test", "role": "test"},
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "open_agent_spec" in error_msg
        assert "missing" in error_msg.lower()
        assert "Example:" in error_msg or "example" in error_msg.lower()

    def test_wrong_type_for_tools(self):
        """Error message should show actual type and expected format."""
        spec_data = {
            "open_agent_spec": "1.0",
            "agent": {"name": "test", "role": "test"},
            "tools": "not a list",  # Wrong type
            "tasks": {},
            "prompts": {"system": "test", "user": "test"},
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "tools" in error_msg
        assert "list" in error_msg or "array" in error_msg
        assert "str" in error_msg  # Shows actual type

    def test_missing_required_agent_field(self):
        """Error message should clearly indicate missing field and provide example."""
        spec_data = {
            "open_agent_spec": "1.0",
            "agent": {"name": "test"},  # Missing 'role'
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "agent.role" in error_msg
        assert "missing" in error_msg.lower()
        assert "example" in error_msg.lower() or "Example:" in error_msg

    def test_tool_references_nonexistent_tool(self):
        """Error message should list available tools."""
        spec_data = {
            "open_agent_spec": "1.0",
            "agent": {"name": "test", "role": "test"},
            "tools": [
                {"id": "tool1", "type": "function", "description": "Tool 1"},
                {"id": "tool2", "type": "function", "description": "Tool 2"},
            ],
            "tasks": {
                "bad_task": {
                    "tool": "nonexistent_tool",  # References non-existent tool
                    "input": {},
                    "output": {},
                }
            },
            "prompts": {"system": "test", "user": "test"},
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "nonexistent_tool" in error_msg
        assert "tool1" in error_msg  # Shows available tools
        assert "tool2" in error_msg
        assert "Available" in error_msg or "available" in error_msg

    def test_wrong_type_for_task_input(self):
        """Error message should show actual type and provide example."""
        spec_data = {
            "open_agent_spec": "1.0",
            "agent": {"name": "test", "role": "test"},
            "tools": [],
            "tasks": {
                "task1": {
                    "input": "not a dict",  # Wrong type
                    "output": {},
                }
            },
            "prompts": {"system": "test", "user": "test"},
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "tasks.task1.input" in error_msg
        assert "dictionary" in error_msg or "object" in error_msg
        assert "str" in error_msg  # Shows actual type
        assert "example" in error_msg.lower() or "Example:" in error_msg

    def test_missing_agent_section(self):
        """Error message should explain how to add the section."""
        spec_data = {
            "open_agent_spec": "1.0",
            # Missing 'agent' section
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "agent" in error_msg
        assert "Missing" in error_msg or "missing" in error_msg
        assert "name:" in error_msg
        assert "role:" in error_msg

    def test_array_item_error_shows_index(self):
        """Error message should show which array item failed."""
        spec_data = {
            "open_agent_spec": "1.0",
            "agent": {"name": "test", "role": "test"},
            "tools": [
                {"id": "tool1", "type": "function", "description": "Good tool"},
                {"id": 123, "type": "function", "description": "Bad tool"},  # id is not string
            ],
            "tasks": {},
            "prompts": {"system": "test", "user": "test"},
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "tools[1]" in error_msg or "tool 1" in error_msg  # Shows index
        assert "id" in error_msg
        assert "int" in error_msg  # Shows actual type


class TestErrorMessageFormat:
    """Test that error messages follow a consistent format."""

    def test_error_includes_field_path(self):
        """All field errors should include the full field path."""
        spec_data = {
            "open_agent_spec": "1.0",
            "agent": {"name": 123, "role": "test"},  # name is wrong type
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        # Should include path like "agent.name" or "Field 'agent.name'"
        assert "agent.name" in error_msg or "agent" in error_msg and "name" in error_msg

    def test_error_shows_actual_type(self):
        """Errors should show what type was actually provided."""
        spec_data = {
            "open_agent_spec": 123,  # Should be string
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        assert "int" in error_msg or "123" in error_msg  # Shows actual type

    def test_error_provides_example(self):
        """Errors should provide an example of correct format."""
        spec_data = {
            "open_agent_spec": "",  # Empty string
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec_data)
        
        error_msg = str(exc_info.value)
        # Should include example format
        assert "example" in error_msg.lower() or '"' in error_msg  # Has quoted example
