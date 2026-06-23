"""Console entry point — run the FDTDMEX MCP discovery server over stdio.

Kept trivial so ``server.py`` stays importable in tests. Launched by name
(``fdtdmex-mcp``) or as ``python -m fdtdmex_mcp``.
"""

from __future__ import annotations

from .server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
