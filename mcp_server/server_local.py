"""
Domain KB — Local-only MCP Server

Same tools as server.py but hardcoded to the ChromaDB backend.
No R2R, no Docker — just pip install chromadb.

Usage in .mcp.json:
  {
    "mcpServers": {
      "domain-kb": {
        "command": "/path/to/.venv/bin/python",
        "args": ["/path/to/mcp_server/server_local.py"],
        "env": {
          "LOCAL_KB_DIR": "/path/to/chroma_db"
        }
      }
    }
  }
"""

import os

# Force local backend before importing server (which reads KB_BACKEND at import time)
os.environ["KB_BACKEND"] = "local"

from mcp_server.server import mcp  # noqa: E402

if __name__ == "__main__":
    mcp.run()
