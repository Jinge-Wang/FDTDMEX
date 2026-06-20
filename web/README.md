# web/ — Web UI (WS-D)

Locally-hostable web front end for the agentic workflow, with a Lumerical-like interactive 3D editor (click to select objects; tabbed panels to switch boundary conditions, materials, source/ detector properties; field overlays).

Rendering options (browser-capable): **plotly** (native web, quickest), **pyvista via trame** (server-side VTK → VTK.js, keeps the Python scene model), or **three.js/react-three-fiber** (most control). Start with plotly/pyvista-trame; reserve three.js for a polished editor.

Talks to the MCP server / config layer (`../server`, `fdtdmex.io`). See [../docs/mcp-and-ui.md](../docs/mcp-and-ui.md). Status: placeholder (no scaffold yet).
