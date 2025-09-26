# mcp-obsidian

An MCP (Model Context Protocol) server for semantic search in Obsidian vaults using 
embedded ChromaDB vector storage.

## Features

- 🔍 Semantic search across your Obsidian vaults using vector embeddings
- 📁 Support for multiple vault configurations
- 🔄 Real-time file watching and automatic index updates
- 🚀 Fast search with ChromaDB backend
- 🔧 Works as both MCP server and CLI tool

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

### Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Install mcp-obsidian

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mcp-obsidian.git
cd mcp-obsidian
```

2. Create and activate a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
uv pip install -e .
```

This will install all dependencies including:
- questionary (interactive CLI)
- chromadb (vector database)
- langchain-text-splitters (document chunking)
- fastmcp (MCP server framework)
- watchdog (file system monitoring)

## Configuration

### Initial Setup

Configure your Obsidian vaults:

```bash
mcp-obsidian configure
```

This interactive command will:
1. Prompt you to select vault directories
2. Name each vault for easy reference
3. Store configuration in `~/.mcp-obsidian/config.json`

### Manual Configuration

You can also manually edit `~/.mcp-obsidian/config.json`:

```json
{
  "vaults": [
    {
      "name": "Personal Notes",
      "path": "/path/to/your/obsidian/vault"
    },
    {
      "name": "Work Docs",
      "path": "/path/to/another/vault"
    }
  ]
}
```

## Usage

### As an MCP Server

Run the server for use with MCP-compatible clients:

```bash
mcp-obsidian
```

The server exposes the following tools:
- `search_notes`: Search across all configured vaults with semantic matching
- `search_vault`: Search within a specific vault

### CLI Usage

Search directly from the command line:

```bash
# Search all vaults
mcp-obsidian search "your search query"

# Search a specific vault
mcp-obsidian search "your search query" --vault "Personal Notes"

# Reconfigure vaults
mcp-obsidian configure

# Rebuild search index
mcp-obsidian index
```

### Integration with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "mcp-obsidian"
    }
  }
}
```

## How It Works

1. **Indexing**: The server reads all markdown files from configured vaults and creates vector embeddings using ChromaDB
2. **Chunking**: Large documents are split into smaller chunks using recursive character splitting for better search granularity
3. **Search**: Queries are converted to embeddings and matched against the document database using cosine similarity
4. **File Watching**: The server monitors vault directories for changes and automatically updates the index

## License

MIT License - see LICENSE file for details
