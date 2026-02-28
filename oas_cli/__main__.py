# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under AGPL-3.0 with Additional Terms
# See LICENSE for details on attribution, naming, and branding restrictions.

"""Allow running the CLI as python -m oas_cli (e.g. in CI where 'oas' may not be on PATH)."""

from .main import app

if __name__ == "__main__":
    app()
