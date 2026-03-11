# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""CLI-facing exceptions with actionable messages."""


class AgentGenerationError(RuntimeError):
    """Raised when agent code generation fails (e.g. missing template, bad spec data)."""

    def __init__(
        self,
        message: str,
        *,
        template_path: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.template_path = template_path
        self.__cause__ = cause
