"""Agent runner -- executes an OA Spec agent against a task.

This is the adapter between the orchestration layer and the existing
``oas_cli.runner`` module.  It loads a spec once and can execute any
of its tasks repeatedly.

The interface is deliberately minimal so it can be replaced by a
remote runner, a Celery worker, a Temporal activity, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class AgentRunner:
    """Execute OA Spec tasks from a loaded spec.

    This is the default in-process implementation.  For distributed
    execution, subclass and override ``run_task``.
    """

    def __init__(self, spec_path: str) -> None:
        self.spec_path = spec_path
        self._spec_data = self._load(spec_path)

    @property
    def agent_name(self) -> str:
        return (self._spec_data.get("agent") or {}).get("name", "unknown")

    @property
    def agent_role(self) -> str:
        return (self._spec_data.get("agent") or {}).get("role", "unknown")

    @property
    def task_names(self) -> list[str]:
        return list((self._spec_data.get("tasks") or {}).keys())

    @property
    def spec_data(self) -> Dict[str, Any]:
        return self._spec_data

    def run_task(
        self,
        task_name: str,
        input_data: Dict[str, Any],
        override_system: Optional[str] = None,
        override_user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single task and return the result envelope.

        Returns a dict with at least ``output`` on success or
        ``error`` on failure.
        """
        # Import here to avoid circular deps and keep the interface importable
        # without a full runtime install.
        from oas_cli.runner import run_task_from_spec, OARunError

        try:
            result = run_task_from_spec(
                self._spec_data,
                task_name=task_name,
                input_data=input_data,
                override_system=override_system,
                override_user=override_user,
            )
            return result
        except OARunError as exc:
            logger.error("Agent %s task %s failed: %s", self.agent_name, task_name, exc)
            return {"error": str(exc), "code": exc.code, "stage": exc.stage}
        except Exception as exc:
            logger.error("Agent %s task %s unexpected error: %s", self.agent_name, task_name, exc)
            return {"error": str(exc)}

    @staticmethod
    def _load(spec_path: str) -> Dict[str, Any]:
        path = Path(spec_path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Spec at {spec_path} did not parse to a dict")
        return data
