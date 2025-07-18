# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under AGPL-3.0 with Additional Terms
# See LICENSE for details on attribution, naming, and branding restrictions.

"""Code generation utilities and framework for Open Agent Spec."""

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Set, Union, Optional
from jinja2 import Environment, FileSystemLoader, meta


class PythonCodeSerializer:
    """Utility for converting Python data structures to code strings."""

    @staticmethod
    def format_value(value: Any) -> str:
        """Convert a Python value to its code representation."""
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            items = [PythonCodeSerializer.format_value(x) for x in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, dict):
            items = [
                f'"{k}": {PythonCodeSerializer.format_value(v)}'
                for k, v in value.items()
            ]
            return f"{{{', '.join(items)}}}"
        else:
            return str(value)

    @staticmethod
    def dict_to_python_code(data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dictionary to properly formatted Python code."""
        if not data:
            return "{}"

        indent_str = "    " * indent
        items = []

        for key, value in data.items():
            if isinstance(value, dict):
                nested_dict = PythonCodeSerializer.dict_to_python_code(
                    value, indent + 1
                )
                items.append(f'{indent_str}    "{key}": {nested_dict}')
            else:
                formatted_value = PythonCodeSerializer.format_value(value)
                items.append(f'{indent_str}    "{key}": {formatted_value}')

        return "{\n" + ",\n".join(items) + f"\n{indent_str}}}"

    @staticmethod
    def list_to_python_code(data: List[str], indent: int = 0) -> str:
        """Convert list of strings to Python list code."""
        if not data:
            return "[]"

        items = [f'"{item}"' for item in data]
        return "[" + ", ".join(items) + "]"


class TemplateVariableParser:
    """Utility for parsing template variables from strings."""

    @staticmethod
    def extract_jinja_variables(template_str: str) -> Set[str]:
        """Extract variables from Jinja2 template string using proper parsing."""
        try:
            env = Environment()
            ast = env.parse(template_str)
            return meta.find_undeclared_variables(ast)
        except Exception:
            # Fallback to manual parsing if Jinja2 parsing fails
            return TemplateVariableParser._manual_extract_variables(template_str)

    @staticmethod
    def _manual_extract_variables(template_str: str) -> Set[str]:
        """Manual extraction of {{variable}} patterns as fallback."""
        import re

        pattern = r"\{\{\s*([^}]+)\s*\}\}"
        matches = re.findall(pattern, template_str)
        variables = set()

        for match in matches:
            # Handle dot notation like input.name -> extract base variable
            if "." in match:
                base_var = match.split(".")[0]
                variables.add(base_var)
            else:
                variables.add(match.strip())

        return variables


class CodeGenerator:
    """Main code generation framework using templates."""

    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        if template_dir is None:
            # Default to templates directory relative to this file
            current_dir = Path(__file__).parent
            template_dir = current_dir / "templates"

        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.serializer = PythonCodeSerializer()
        self.variable_parser = TemplateVariableParser()

    def generate_from_template(self, template_name: str, **data: Any) -> str:
        """Generate code from a template file."""
        template = self.env.get_template(template_name)
        return template.render(**data)

    def generate_python_dict(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Generate Python dictionary code."""
        return self.serializer.dict_to_python_code(data, indent)

    def generate_python_list(self, data: List[str], indent: int = 0) -> str:
        """Generate Python list code."""
        return self.serializer.list_to_python_code(data, indent)

    def format_value(self, value: Any) -> str:
        """Format a single value as Python code."""
        return self.serializer.format_value(value)

    def ensure_template_exists(self, template_name: str, default_content: str) -> None:
        """Ensure a template file exists, creating it with default content if not."""
        template_path = self.template_dir / template_name
        if not template_path.exists():
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text(default_content)

    def indent_text(self, text: str, indent: int = 1) -> str:
        """Indent text by the specified number of levels (4 spaces each)."""
        return textwrap.indent(text, "    " * indent)
