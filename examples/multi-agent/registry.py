"""Agent registry -- tracks which agents are available and what they can do.

Each entry records the agent's ID, the OA spec file it was loaded from,
its declared role, and a reference to whatever runner will execute it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEntry:
    """A registered agent."""

    id: str
    role: str
    spec_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "spec_path": self.spec_path,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
        }


class AgentRegistry:
    """In-memory agent registry."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentEntry] = {}

    def register(
        self,
        agent_id: str,
        role: str,
        spec_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEntry:
        entry = AgentEntry(
            id=agent_id,
            role=role,
            spec_path=spec_path,
            metadata=metadata or {},
        )
        self._agents[agent_id] = entry
        return entry

    def unregister(self, agent_id: str) -> bool:
        return self._agents.pop(agent_id, None) is not None

    def get(self, agent_id: str) -> AgentEntry | None:
        return self._agents.get(agent_id)

    def find_by_role(self, role: str) -> list[AgentEntry]:
        return [a for a in self._agents.values() if a.role == role]

    def all_agents(self) -> list[AgentEntry]:
        return list(self._agents.values())

    def summary(self) -> dict[str, Any]:
        roles: dict[str, int] = {}
        for a in self._agents.values():
            roles[a.role] = roles.get(a.role, 0) + 1
        return {
            "total": len(self._agents),
            "by_role": roles,
        }
