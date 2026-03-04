from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import yaml

import oas_cli
from oas_cli.generators import (
    generate_agent_code,
    generate_env_example,
    generate_prompt_template,
    generate_readme,
    generate_requirements,
)
from oas_cli.validators import (
    validate_spec as _validate_spec_data,
)
from oas_cli.validators import (
    validate_with_json_schema,
)


def _build_demo_snippet(spec_data: dict[str, Any]) -> str:
    """Build a minimal, spec-driven Python snippet for the first task."""
    tasks = spec_data.get("tasks") or {}
    first_task_name = next(iter(tasks), "greet")
    task = tasks.get(first_task_name, {})
    model_name = (spec_data.get("intelligence") or {}).get("model", "gpt-4")
    prompts = spec_data.get("prompts") or {}
    system_text = str(prompts.get("system", "")).strip()
    user_tpl = str(prompts.get("user", "{{ name }}")).strip()

    # Input: param names from task.input
    input_props = (task.get("input") or {}).get("properties") or {}
    input_required = (task.get("input") or {}).get("required") or list(
        input_props.keys()
    )
    param_names = (
        [k for k in input_required if k in input_props]
        or list(input_props.keys())
        or ["name"]
    )
    param_decls = ", ".join(f"{p}: str" for p in param_names)
    # User prompt: replace {{ name }} / {{ topic }} etc. with f-string placeholders
    user_content = user_tpl
    for p in param_names:
        user_content = user_content.replace("{{ " + p + " }}", "{" + p + "}")
    user_prompt_line = f'user_prompt = f"{user_content}"'

    # Output: Pydantic model from task.output
    output_props = (task.get("output") or {}).get("properties") or {
        "response": {"type": "string"}
    }
    output_required = (task.get("output") or {}).get("required") or list(
        output_props.keys()
    )
    class_name = first_task_name.replace("-", " ").title().replace(" ", "") + "Output"
    lines = []
    for key in output_required or output_props:
        if key not in output_props:
            continue
        prop = output_props[key]
        if isinstance(prop, dict) and prop.get("type") == "array":
            lines.append(f"    {key}: list[str]")
        else:
            lines.append(f"    {key}: str")
    model_fields = "\n".join(lines) if lines else "    response: str"

    # Return: one keyword per output field
    return_kwargs = []
    for key in output_required or output_props:
        if key not in output_props:
            continue
        prop = output_props.get(key) or {}
        if isinstance(prop, dict) and prop.get("type") == "array":
            return_kwargs.append(f'{key}=data.get("{key}", [])')
        else:
            return_kwargs.append(f'{key}=data.get("{key}", raw)')
    return_stmt = (
        f"return {class_name}({', '.join(return_kwargs)})"
        if return_kwargs
        else 'return {class_name}(response=data.get("response", raw))'
    )

    snippet = f'''# Generated from Open Agent Spec — prompts and output schema from YAML
import json
from openai import OpenAI
from pydantic import BaseModel


class {class_name}(BaseModel):
{model_fields}


client = OpenAI()


def {first_task_name}({param_decls}):
    system_prompt = """{system_text}"""
    {user_prompt_line}
    response = client.chat.completions.create(
        model="{model_name}",
        messages=[
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": user_prompt}},
        ],
    )
    raw = response.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {{}}
    {return_stmt}
'''
    return snippet


def _validate_spec_file(spec_path: Path) -> tuple[dict[str, Any], str, str]:
    """
    Load and validate a spec file using the installed `open-agent-spec` package.

    This mirrors the behaviour of `oas_cli.core.validate_spec_file` but avoids
    importing the `core` module directly (older PyPI versions may not expose it).
    """
    try:
        with spec_path.open(encoding="utf-8") as f:
            spec_data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as err:
        raise ValueError(f"Invalid spec path or YAML: {err}") from err

    # Older/open-source validators currently expect a behavioural_contract block
    # and enforce that behavioural_contract.version is a string. For the web
    # demo we want a contract-free spec, so we inject a minimal default only
    # for validation/generation here (not shown in the YAML).
    if "behavioural_contract" not in spec_data:
        spec_data["behavioural_contract"] = {
            "version": "0.1.2",
            "description": "",
        }

    schema_path = Path(oas_cli.__file__).parent / "schemas" / "oas-schema.json"
    validate_with_json_schema(spec_data, str(schema_path))
    agent_name, class_name = _validate_spec_data(spec_data)
    return spec_data, agent_name, class_name


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw_body or "{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON body"})
                return

            yaml_str = payload.get("yaml")
            if not isinstance(yaml_str, str) or not yaml_str.strip():
                self._send_json(
                    400, {"error": "Missing or empty 'yaml' in request body"}
                )
                return

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                spec_path = tmp / "spec.yaml"
                output_dir = tmp / "out"
                spec_path.write_text(yaml_str, encoding="utf-8")

                try:
                    spec_data, agent_name, class_name = _validate_spec_file(spec_path)
                except ValueError as e:  # validation or YAML error
                    self._send_json(422, {"error": str(e)})
                    return

                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Use the real CLI generator to produce the full project
                    # scaffold (README, requirements, prompts, etc.).
                    generate_agent_code(output_dir, spec_data, agent_name, class_name)
                    generate_readme(output_dir, spec_data)
                    generate_requirements(output_dir, spec_data)
                    generate_env_example(output_dir, spec_data)
                    generate_prompt_template(output_dir, spec_data)
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
                    self._send_json(422, {"error": f"Generation failed: {e}"})
                    return

                # Collect generated file contents for JSON response. For the
                # demo we intentionally return a frictionless, minimal agent
                # that does not depend on DACP, even though the on-disk
                # project includes the full scaffold.
                result = {}
                result["agentPy"] = _build_demo_snippet(spec_data)
                readme = output_dir / "README.md"
                if readme.exists():
                    result["readme"] = readme.read_text(encoding="utf-8")
                req = output_dir / "requirements.txt"
                if req.exists():
                    result["requirementsTxt"] = req.read_text(encoding="utf-8")
                env_example = output_dir / ".env.example"
                if env_example.exists():
                    result["envExample"] = env_example.read_text(encoding="utf-8")

                prompts_dir = output_dir / "prompts"
                if prompts_dir.is_dir():
                    prompts: dict[str, str] = {}
                    for f in prompts_dir.iterdir():
                        if f.is_file() and f.suffix in (".jinja2", ".j2"):
                            prompts[f.name] = f.read_text(encoding="utf-8")
                    if prompts:
                        result["prompts"] = prompts

                self._send_json(200, result)
        except Exception as e:  # pragma: no cover - defensive catch-all
            self._send_json(500, {"error": str(e) or "Internal server error"})


# Vercel Python runtime expects a symbol named `handler` that is a
# BaseHTTPRequestHandler subclass; expose the class here.
handler = Handler
