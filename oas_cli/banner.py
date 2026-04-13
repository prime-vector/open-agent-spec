"""OA CLI banner and branding assets."""

# Compact bot-face banner. Renders cleanly at 80+ columns in any modern terminal.
# The ◈ glyph is the OA "agent node" icon used consistently across CLI surfaces.

BANNER = r"""
  ╔═══╗
  ║◈ ◈║  Open Agent Spec
  ╚═╤═╝  Agents as code.
    ╧
"""

# Keep the old name so existing callers in main.py still work until
# they are migrated to the new ui module.
ASCII_TITLE = BANNER
