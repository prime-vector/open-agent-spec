import json
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from oas_cli.core import validate_spec_file
from oas_cli.generators import (
    generate_agent_code,
    generate_env_example,
    generate_prompt_template,
    generate_readme,
    generate_requirements,
)


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
                    spec_data, agent_name, class_name = validate_spec_file(spec_path)
                except ValueError as e:  # validation or YAML error
                    self._send_json(422, {"error": str(e)})
                    return

                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    generate_agent_code(output_dir, spec_data, agent_name, class_name)
                    generate_readme(output_dir, spec_data)
                    generate_requirements(output_dir, spec_data)
                    generate_env_example(output_dir, spec_data)
                    generate_prompt_template(output_dir, spec_data)
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
                    self._send_json(422, {"error": f"Generation failed: {e}"})
                    return

                # Collect generated file contents for JSON response
                result: dict[str, object] = {}
                agent_py = output_dir / "agent.py"
                if agent_py.exists():
                    result["agentPy"] = agent_py.read_text(encoding="utf-8")
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

