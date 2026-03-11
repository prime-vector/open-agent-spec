# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Allow running the CLI as python -m oas_cli (e.g. in CI where 'oa' is not on PATH)."""

from .main import app

if __name__ == "__main__":
    app()
