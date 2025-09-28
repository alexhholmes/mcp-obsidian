#!/usr/bin/env python3
import argparse
import json
import sys
import hashlib
import os
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime, timedelta

# Disable ChromaDB telemetry before importing to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import questionary
from questionary import Style
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastmcp import FastMCP
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


# Configuration paths
CONFIG_DIR = Path.home() / ".mcp-obsidian"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ChromaDB settings
CHROMA_DB_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "obsidian_notes"
CHROMA_COLLECTION_DESC = "Obsidian vault notes"

# Text chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Search settings
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 20
PREVIEW_LENGTH = 500

# File monitoring settings
MONITOR_UPDATE_INTERVAL = 10  # seconds
MONITOR_DEBOUNCE_SECONDS = 30  # seconds

# Create FastMCP server instance
mcp = FastMCP("mcp-obsidian")

# Global ChromaDB client and collection
chroma_client: Optional[chromadb.Client] = None
chroma_collection: Optional[chromadb.Collection] = None

# Lock for updating global ChromaDB references atomically
import threading
chroma_ref_lock = threading.Lock()

custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
    ('disabled', 'fg:#858585 italic')
])


def add_vault_path():
    """Add a new vault path to the configuration"""
    existing_config = load_config()
    vaults = existing_config.get("vaults", [])
    existing_paths = {v["path"] for v in vaults}

    while True:
        # Ask for vault path with path autocomplete
        vault_path = questionary.path(
            "Enter Obsidian vault path:",
            only_directories=True,
            style=custom_style
        ).ask()

        if not vault_path:
            break

        path = Path(vault_path).expanduser().resolve()

        # Check if already configured
        if str(path) in existing_paths:
            print(f"⚠️  Vault already configured: {path}\n", file=sys.stderr)
            continue

        # Validate path exists
        if not path.exists():
            if not questionary.confirm(
                f"Path does not exist: {path}\nAdd anyway?",
                default=False,
                style=custom_style
            ).ask():
                continue

        # Ask for vault name
        vault_name = questionary.text(
            "Vault name:",
            default=path.name,
            style=custom_style
        ).ask()

        vaults.append({
            "name": vault_name,
            "path": str(path)
        })
        existing_paths.add(str(path))

        print(f"✅ Added: {vault_name} → {path}\n", file=sys.stderr)

        if not questionary.confirm(
            "Add another vault?",
            default=False,
            style=custom_style
        ).ask():
            break

    if vaults:
        save_config(vaults)
        print(f"\n📚 Total configured vaults: {len(vaults)}", file=sys.stderr)


def list_vault_paths():
    """List all configured vault paths"""
    config = load_config()
    vaults = config.get("vaults", [])

    if not vaults:
        print("📭 No vaults configured yet.", file=sys.stderr)
        print("Use 'Add Vault Path' to configure your first vault.", file=sys.stderr)
        return

    print(f"📚 Configured Vaults ({len(vaults)}):\n", file=sys.stderr)
    for i, vault in enumerate(vaults, 1):
        print(f"  {i}. {vault['name']}", file=sys.stderr)
        print(f"     📁 {vault['path']}", file=sys.stderr)
        if i < len(vaults):
            print(file=sys.stderr)


def remove_vault_path():
    """Remove a vault path from the configuration"""
    config = load_config()
    vaults = config.get("vaults", [])

    if not vaults:
        print("📭 No vaults configured yet.", file=sys.stderr)
        print("Use 'Add Vault Path' to configure your first vault.", file=sys.stderr)
        return

    print(f"📚 Select vault to remove:\n", file=sys.stderr)

    # Create choices list with vault information
    choices = []
    for i, vault in enumerate(vaults, 1):
        choice_text = f"{vault['name']} → {vault['path']}"
        choices.append({"name": choice_text, "value": i - 1})

    # Add cancel option
    choices.append({"name": "Cancel", "value": -1})

    selected = questionary.select(
        "Which vault would you like to remove?",
        choices=choices,
        style=custom_style
    ).ask()

    if selected == -1 or selected is None:
        print("Removal cancelled.", file=sys.stderr)
        return

    # Get the vault to remove
    vault_to_remove = vaults[selected]

    # Confirm deletion
    if questionary.confirm(
        f"Are you sure you want to remove '{vault_to_remove['name']}'?",
        default=False,
        style=custom_style
    ).ask():
        # Remove the vault
        vaults.pop(selected)

        # Save updated configuration
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config = {
            "vaults": vaults,
            "version": "1.0"
        }

        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Removed: {vault_to_remove['name']} → {vault_to_remove['path']}", file=sys.stderr)
        print(f"💾 Configuration updated: {CONFIG_FILE}", file=sys.stderr)
    else:
        print("Removal cancelled.", file=sys.stderr)


def configure_claude_desktop():
    """Configure Claude Desktop to use mcp-obsidian"""
    import platform

    # Determine the Claude Desktop config path based on platform
    system = platform.system()
    if system == "Darwin":  # macOS
        claude_config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        claude_config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    else:  # Linux
        claude_config_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

    print("🤖 Configuring Claude Desktop for mcp-obsidian\n", file=sys.stderr)

    # Check if Claude config file exists
    if not claude_config_path.exists():
        print(f"📝 Creating Claude Desktop configuration at:\n   {claude_config_path}\n", file=sys.stderr)
        claude_config_path.parent.mkdir(parents=True, exist_ok=True)
        claude_config = {}
    else:
        print(f"📝 Found Claude Desktop configuration at:\n   {claude_config_path}\n", file=sys.stderr)
        try:
            with open(claude_config_path, 'r') as f:
                claude_config = json.load(f)
        except json.JSONDecodeError:
            print("⚠️  Existing configuration is invalid. Creating new configuration.\n", file=sys.stderr)
            claude_config = {}

    # Ensure mcpServers section exists
    if "mcpServers" not in claude_config:
        claude_config["mcpServers"] = {}

    # Check if obsidian is already configured
    if "obsidian" in claude_config["mcpServers"]:
        existing = claude_config["mcpServers"]["obsidian"]
        print("⚠️  MCP Obsidian is already configured in Claude Desktop:", file=sys.stderr)
        print(f"   Command: {existing.get('command', 'Not set')}", file=sys.stderr)

        if not questionary.confirm(
            "\nDo you want to update the existing configuration?",
            default=True,
            style=custom_style
        ).ask():
            print("Configuration cancelled.", file=sys.stderr)
            return

    # Add or update the obsidian configuration
    claude_config["mcpServers"]["obsidian"] = {
        "command": "mcp-obsidian"
    }

    # Save the updated configuration
    try:
        with open(claude_config_path, 'w') as f:
            json.dump(claude_config, f, indent=2)

        print("\n✅ Claude Desktop configuration updated successfully!", file=sys.stderr)
        print("\n📋 Configuration added:", file=sys.stderr)
        print('   "obsidian": {', file=sys.stderr)
        print('     "command": "mcp-obsidian"', file=sys.stderr)
        print('   }', file=sys.stderr)
        print("\n🔄 Please restart Claude Desktop for the changes to take effect.", file=sys.stderr)

        # Check if vaults are configured
        config = load_config()
        vaults = config.get("vaults", [])
        if not vaults:
            print("\n⚠️  No vaults configured yet!", file=sys.stderr)
            print("   Run 'mcp-obsidian configure' and select 'Add Vault Path' to add your Obsidian vaults.", file=sys.stderr)

    except Exception as e:
        print(f"❌ Failed to update Claude Desktop configuration: {e}", file=sys.stderr)
        print("\nYou can manually add the following to your Claude Desktop configuration:", file=sys.stderr)
        print(f"File: {claude_config_path}", file=sys.stderr)
        print('\n{', file=sys.stderr)
        print('  "mcpServers": {', file=sys.stderr)
        print('    "obsidian": {', file=sys.stderr)
        print('      "command": "mcp-obsidian"', file=sys.stderr)
        print('    }', file=sys.stderr)
        print('  }', file=sys.stderr)
        print('}', file=sys.stderr)


def save_config(vaults):
    """Save the configuration to file"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "vaults": vaults,
        "version": "1.0"
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"💾 Configuration saved to: {CONFIG_FILE}", file=sys.stderr)


def configure_exclude_paths():
    """Configure paths to exclude from indexing (glob patterns)"""
    config = load_config()
    exclude_patterns = config.get("exclude_patterns", [])

    print("🚫 Configure Excluded Paths\n", file=sys.stderr)
    print("Use glob patterns to exclude files/folders from indexing.", file=sys.stderr)
    print("Examples:", file=sys.stderr)
    print("  • **/Archive/**  (all Archive folders)", file=sys.stderr)
    print("  • **/Private/*   (Private folder contents)", file=sys.stderr)
    print("  • **/*.tmp.md    (temporary markdown files)", file=sys.stderr)
    print("  • **/Trash/**    (Trash/Recycle folders)\n", file=sys.stderr)

    while True:
        # Show current patterns if any
        if exclude_patterns:
            print("📋 Current exclude patterns:", file=sys.stderr)
            for i, pattern in enumerate(exclude_patterns, 1):
                print(f"   {i}. {pattern}", file=sys.stderr)
            print(file=sys.stderr)
        else:
            print("📭 No exclude patterns configured yet.\n", file=sys.stderr)

        # Ask what to do
        action = questionary.select(
            "What would you like to do?",
            choices=[
                {"name": "Add Pattern", "value": "add"},
                {"name": "Remove Pattern", "value": "remove"},
                {"name": "Clear All Patterns", "value": "clear"},
                {"name": "Back to Main Menu", "value": "back"}
            ],
            style=custom_style
        ).ask()

        if action == "back" or action is None:
            break

        elif action == "add":
            # Add new pattern
            pattern = questionary.text(
                "Enter glob pattern to exclude:",
                instruction="(e.g., **/Archive/** or **/*.tmp.md)",
                style=custom_style
            ).ask()

            if pattern and pattern not in exclude_patterns:
                exclude_patterns.append(pattern)
                print(f"✅ Added pattern: {pattern}", file=sys.stderr)

                # Save updated config
                config["exclude_patterns"] = exclude_patterns
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"💾 Configuration saved", file=sys.stderr)
            elif pattern in exclude_patterns:
                print(f"⚠️  Pattern already exists: {pattern}", file=sys.stderr)

        elif action == "remove" and exclude_patterns:
            # Remove a pattern
            choices = [{"name": pattern, "value": i} for i, pattern in enumerate(exclude_patterns)]
            choices.append({"name": "Cancel", "value": -1})

            selected = questionary.select(
                "Select pattern to remove:",
                choices=choices,
                style=custom_style
            ).ask()

            if selected != -1 and selected is not None:
                removed_pattern = exclude_patterns.pop(selected)
                print(f"✅ Removed pattern: {removed_pattern}", file=sys.stderr)

                # Save updated config
                config["exclude_patterns"] = exclude_patterns
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"💾 Configuration saved", file=sys.stderr)

        elif action == "clear" and exclude_patterns:
            # Clear all patterns
            if questionary.confirm(
                f"Remove all {len(exclude_patterns)} exclude patterns?",
                default=False,
                style=custom_style
            ).ask():
                exclude_patterns = []
                config["exclude_patterns"] = exclude_patterns
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                print("✅ All exclude patterns cleared", file=sys.stderr)
                print(f"💾 Configuration saved", file=sys.stderr)

        print(file=sys.stderr)  # Add spacing


def configure():
    """Configure Obsidian vault paths"""
    print("🗂️  MCP Obsidian Configuration\n", file=sys.stderr)

    while True:
        try:
            choice = questionary.select(
                "What would you like to do?",
                choices=[
                    {"name": "Add Vault Path", "value": "1", "shortcut_key": "1"},
                    {"name": "List Vault Paths", "value": "2", "shortcut_key": "2"},
                    {"name": "Remove Vault Path", "value": "3", "shortcut_key": "3"},
                    {"name": "Exclude Paths", "value": "4", "shortcut_key": "4"},
                    {"name": "Configure Claude Desktop", "value": "5", "shortcut_key": "5"},
                    {"name": "Exit (^C)", "value": "6", "shortcut_key": "6"}
                ],
                style=custom_style,
                use_shortcuts=True,
                use_arrow_keys=True,
                show_selected=True,
                use_jk_keys=False
            ).ask()
        except KeyboardInterrupt:
            sys.exit(0)  # Exit silently on Ctrl+C

        if choice == "1":
            add_vault_path()
        elif choice == "2":
            list_vault_paths()
        elif choice == "3":
            remove_vault_path()
        elif choice == "4":
            configure_exclude_paths()
        elif choice == "5":
            configure_claude_desktop()
        elif choice == "6" or choice is None:
            break

        print(file=sys.stderr)  # Add spacing between operations


def is_file_excluded(file_path: Path, vault_path: Path) -> bool:
    """Check if a file should be excluded based on configured patterns

    Args:
        file_path: The file path to check
        vault_path: The vault root path (for relative path matching)

    Returns:
        True if the file should be excluded, False otherwise
    """
    import fnmatch

    # Load current configuration to get latest exclude patterns
    config = load_config()
    exclude_patterns = config.get("exclude_patterns", [])

    if not exclude_patterns:
        return False

    # Get relative path from vault root for pattern matching
    try:
        relative_path = file_path.relative_to(vault_path)
        relative_str = str(relative_path).replace('\\', '/')  # Normalize path separators
    except ValueError:
        # If file is not in vault path, use absolute path
        relative_str = str(file_path).replace('\\', '/')

    # Check if file matches any exclude pattern
    for pattern in exclude_patterns:
        # Support ** glob patterns by converting to fnmatch pattern
        # ** means any number of directories
        fnmatch_pattern = pattern.replace('**/', '*/').replace('**', '*')

        # Check both the relative path and the full path
        if fnmatch.fnmatch(relative_str, pattern):
            return True
        if fnmatch.fnmatch(relative_str, fnmatch_pattern):
            return True
        # Also check against each path component for patterns like **/Archive/**
        path_parts = relative_str.split('/')
        for i in range(len(path_parts)):
            partial_path = '/'.join(path_parts[i:])
            if fnmatch.fnmatch(partial_path, pattern.lstrip('**/')):
                return True
            # Check if any directory in the path matches a directory pattern
            if '**/' in pattern:
                dir_pattern = pattern.replace('**/', '').rstrip('/*').rstrip('/**')
                if dir_pattern in path_parts:
                    return True

    return False


def get_markdown_files(vault_path: str) -> List[Path]:
    """Recursively get all markdown files from a vault"""
    vault = Path(vault_path)
    if not vault.exists():
        return []

    markdown_files = []
    # Use rglob for recursive search
    for file_path in vault.rglob("*.md"):
        # Skip hidden directories and files
        if any(part.startswith('.') for part in file_path.parts):
            continue
        # Skip files matching exclude patterns
        if is_file_excluded(file_path, vault):
            continue
        markdown_files.append(file_path)

    return markdown_files


def get_line_boundaries(text: str, chunk_start: int, chunk_end: int) -> Tuple[int, int]:
    """Calculate line numbers for a chunk within the original text"""
    lines_before_chunk = text[:chunk_start].count('\n')
    lines_in_chunk = text[chunk_start:chunk_end].count('\n')

    start_line = lines_before_chunk + 1
    end_line = start_line + lines_in_chunk

    return start_line, end_line


def chunk_markdown_content(file_path: Path, vault_path: Path, vault_name: str) -> List[Dict]:
    """Read and chunk a markdown file with enhanced metadata"""
    try:
        # Check file size before loading
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            print(f"⚠️  Skipping large file ({size_mb:.1f}MB): {file_path.name}", file=sys.stderr)
            return []  # Skip files that are too large

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Calculate actual content size in bytes
        content_size_bytes = len(content.encode('utf-8'))

        # Get file metadata
        file_stats = os.stat(file_path)
        modified_time = int(file_stats.st_mtime)

        # Create file content hash for change detection
        file_hash = hashlib.md5(content.encode()).hexdigest()

        # Get relative path from vault root
        relative_path = file_path.relative_to(vault_path)

        # Get file stem (filename without extension)
        title = file_path.stem

        # Create text splitter for markdown
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS,
            keep_separator=False
        )

        # Split the content and track positions
        chunks = text_splitter.create_documents([content])

        # Create document chunks with metadata
        documents = []
        current_position = 0

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content

            # Find the position of this chunk in the original content
            chunk_start = content.find(chunk_text, current_position)
            chunk_end = chunk_start + len(chunk_text)
            current_position = chunk_end

            # Get line boundaries
            start_line, end_line = get_line_boundaries(content, chunk_start, chunk_end)

            doc_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()

            documents.append({
                "id": doc_id,
                "content": chunk_text,
                "metadata": {
                    "title": title,
                    "source": str(relative_path),
                    "vault": vault_name,
                    "modified": modified_time,
                    "file_hash": file_hash,  # Add hash for change detection
                    "start_line": start_line,
                    "end_line": end_line,
                    "file_path": str(file_path),  # Keep absolute path for reference
                    "file_size_bytes": content_size_bytes,  # Size of the full document
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

        return documents
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return []


def get_files_to_update(collection: chromadb.Collection, vault_files: List[Path]) -> Tuple[List[Path], List[str]]:
    """Determine which files need to be updated based on modification time"""
    files_to_update = []
    ids_to_delete = []

    # Get all existing documents metadata
    try:
        existing_docs = collection.get(include=['metadatas'])
        if not existing_docs['ids']:
            # Empty collection, all files need indexing
            return vault_files, []
    except:
        # Collection doesn't exist or is empty
        return vault_files, []

    # Create a map of file paths to their metadata
    existing_files = {}
    file_to_ids = {}  # Map file paths to their document IDs

    for i, doc_id in enumerate(existing_docs['ids']):
        metadata = existing_docs['metadatas'][i]
        file_path = metadata.get('file_path')
        if file_path:
            if file_path not in file_to_ids:
                file_to_ids[file_path] = []
            file_to_ids[file_path].append(doc_id)

            if file_path not in existing_files:
                existing_files[file_path] = metadata

    # Check each file in the vault
    vault_file_paths = set(str(f) for f in vault_files)

    for file_path in vault_files:
        file_path_str = str(file_path)

        if file_path_str not in existing_files:
            # New file, needs to be indexed
            files_to_update.append(file_path)
        else:
            # Check if file has been modified
            try:
                file_stats = os.stat(file_path)
                current_mtime = int(file_stats.st_mtime)
                stored_mtime = existing_files[file_path_str].get('modified', 0)

                if current_mtime > stored_mtime:
                    # File has been modified
                    files_to_update.append(file_path)
                    # Mark old chunks for deletion
                    if file_path_str in file_to_ids:
                        ids_to_delete.extend(file_to_ids[file_path_str])
            except FileNotFoundError:
                # File no longer exists, will be handled below
                pass

    # Find files that exist in DB but not on disk (deleted files)
    for file_path_str in existing_files:
        if file_path_str not in vault_file_paths:
            # File has been deleted, remove its chunks
            if file_path_str in file_to_ids:
                ids_to_delete.extend(file_to_ids[file_path_str])

    return files_to_update, ids_to_delete


def initialize_vector_store(vaults: List[Dict], force_rebuild: bool = False) -> Tuple[chromadb.Client, chromadb.Collection]:
    """Initialize ChromaDB and incrementally index vault content

    Args:
        vaults: List of vault configurations
        force_rebuild: If True, clear existing database before rebuilding
    """
    # Handle force rebuild under lock
    if force_rebuild:
        with chroma_ref_lock:
            print("🔥 Force rebuilding index (clearing existing data)...", file=sys.stderr)
            # Clear the existing database
            db_path = CONFIG_DIR / CHROMA_DB_DIR
            if db_path.exists():
                import shutil
                shutil.rmtree(db_path)
                print("   Cleared existing index", file=sys.stderr)

            # Clear global references while rebuilding
            global chroma_client, chroma_collection
            chroma_client = None
            chroma_collection = None

    print("🔄 Initializing vector store...", file=sys.stderr)

    # Disable ChromaDB telemetry to avoid errors
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    # Initialize ChromaDB with persistent storage
    db_path = CONFIG_DIR / CHROMA_DB_DIR
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Get or create collection (don't delete existing)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": CHROMA_COLLECTION_DESC}
    )

    print(f"📚 Checking {len(vaults)} vault(s) for updates...", file=sys.stderr)

    total_files_updated = 0
    total_files_deleted = 0
    total_chunks_added = 0
    all_ids_to_delete = []

    for vault in vaults:
        vault_path = Path(vault["path"])
        vault_name = vault["name"]

        print(f"\n🗂️  Checking vault: {vault_name}", file=sys.stderr)
        print(f"   Path: {vault_path}", file=sys.stderr)

        # Get all markdown files recursively
        markdown_files = get_markdown_files(str(vault_path))
        print(f"   Found {len(markdown_files)} markdown files", file=sys.stderr)

        # Determine which files need updating
        files_to_update, ids_to_delete = get_files_to_update(collection, markdown_files)
        all_ids_to_delete.extend(ids_to_delete)

        if not files_to_update and not ids_to_delete:
            print(f"   ✅ Vault is up to date", file=sys.stderr)
            continue

        if ids_to_delete:
            print(f"   Removing {len(set(f.split('_')[0] for f in ids_to_delete if '_' in f))} outdated/deleted files", file=sys.stderr)
            total_files_deleted += len(set(f.split('_')[0] for f in ids_to_delete if '_' in f))

        if files_to_update:
            print(f"   Updating {len(files_to_update)} changed/new files", file=sys.stderr)

            # Process files that need updating
            for idx, file_path in enumerate(files_to_update, 1):
                print(f"   Processing file {idx}/{len(files_to_update)}: {file_path.name}", end="\r", file=sys.stderr)
                documents = chunk_markdown_content(file_path, vault_path, vault_name)

                if documents:
                    # Prepare data for ChromaDB
                    ids = [doc["id"] for doc in documents]
                    contents = [doc["content"] for doc in documents]
                    metadatas = [doc["metadata"] for doc in documents]

                    # Add to collection under lock protection
                    with chroma_ref_lock:
                        collection.add(
                            ids=ids,
                            documents=contents,
                            metadatas=metadatas
                        )

                    total_files_updated += 1
                    total_chunks_added += len(documents)

            # Clear the progress line and show completion
            print(f"   ✅ Updated {len(files_to_update)} files from {vault_name}                    ", file=sys.stderr)

    # Delete outdated chunks if any
    if all_ids_to_delete:
        print(f"\n🗑️  Removing {len(all_ids_to_delete)} outdated chunks...", file=sys.stderr)
        try:
            with chroma_ref_lock:
                collection.delete(ids=all_ids_to_delete)
        except Exception as e:
            print(f"   Warning: Could not delete some chunks: {e}", file=sys.stderr)

    total_docs = collection.count()

    print(f"\n✅ Indexing complete!", file=sys.stderr)
    print(f"   Files updated: {total_files_updated}", file=sys.stderr)
    print(f"   Files removed: {total_files_deleted}", file=sys.stderr)
    print(f"   Chunks added: {total_chunks_added}", file=sys.stderr)
    print(f"   Total database size: {total_docs} documents", file=sys.stderr)

    return client, collection


def get_full_path_from_metadata(metadata: Dict) -> str:
    """
    Get the full file path from search result metadata.

    Args:
        metadata: The metadata dictionary from search results

    Returns:
        The full absolute path to the file
    """
    # Try to get full path from stored file_path first
    full_path = metadata.get('file_path')

    # If not available, construct it from vault config
    if not full_path:
        vault_name = metadata.get('vault', 'Unknown vault')
        relative_path = metadata.get('source', 'Unknown')

        if vault_name != 'Unknown vault' and relative_path != 'Unknown':
            config = load_config()
            vaults = config.get("vaults", [])

            for vault in vaults:
                if vault["name"] == vault_name:
                    vault_path = vault["path"]
                    full_path = str(Path(vault_path) / relative_path)
                    break

    # Fallback to relative path if we couldn't construct full path
    if not full_path:
        full_path = metadata.get('source', 'Unknown')

    return full_path


def parse_contains_query(query: str) -> Dict:
    """
    Parse query with quoted phrases into semantic and exact phrase components.

    Supports:
    - "exact phrase" - Must contain this exact phrase
    - -"excluded phrase" - Must not contain this phrase
    - Regular text - For semantic search
    """
    required_phrases = []
    excluded_phrases = []

    # Pattern to find quoted strings with optional negative prefix
    pattern = r'(-?)"([^"]+)"'

    # Extract all quoted phrases
    remaining = query
    for match in re.finditer(pattern, query):
        full_match = match.group(0)
        is_negative = match.group(1) == '-'
        phrase = match.group(2)

        if is_negative:
            excluded_phrases.append(phrase)
        else:
            required_phrases.append(phrase)

        # Remove from query
        remaining = remaining.replace(full_match, '', 1)

    # Clean up remaining semantic query
    semantic_query = ' '.join(remaining.split()).strip()

    return {
        'semantic_query': semantic_query,
        'required_phrases': required_phrases,
        'excluded_phrases': excluded_phrases
    }


@mcp.tool
async def search(
    query: str,
    vault_filter: Optional[str] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = "desc",
    limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    """
    Unified search with semantic, exact phrase, and temporal filtering.

    Query syntax:
    - "exact phrase" - Must contain this exact phrase (case-sensitive)
    - -"excluded phrase" - Must not contain this phrase
    - Regular text - Semantic similarity search

    Args:
        query: The search query with optional quoted phrases
        vault_filter: Optional filter results to a specific vault name
        since_date: Optional start date in YYYY-MM-DD format (inclusive)
        until_date: Optional end date in YYYY-MM-DD format (inclusive)
        sort_by: Sort results by "relevance" (default), "date", "title", or "path"
        sort_order: Sort order "desc" (default) or "asc"
        limit: Maximum number of results to return (default: 10, max: 20)

    Note: Results include file size metadata to help determine if retrieving
    the full document is feasible. Files larger than 10MB are not indexed.

    Examples:
    - 'PKM "second brain"' - Semantic search for PKM containing "second brain"
    - 'meeting -"cancelled"' with since_date="2024-01-01" - Recent meeting notes
    - '"daily note"' with sort_by="date" - Daily notes sorted by modification date
    - 'python' with sort_by="title", sort_order="asc" - Python notes sorted A-Z
    """
    global chroma_collection

    # Cap limit at max
    limit = min(limit, MAX_SEARCH_LIMIT)

    # Parse the query for quoted phrases
    parsed = parse_contains_query(query)

    # Parse dates and convert to timestamps if provided
    since_timestamp = None
    until_timestamp = None

    try:
        if since_date:
            since_dt = datetime.strptime(since_date, "%Y-%m-%d")
            since_timestamp = int(since_dt.timestamp())

        if until_date:
            # Add 23:59:59 to include the entire day
            until_dt = datetime.strptime(until_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
            until_timestamp = int(until_dt.timestamp())
    except ValueError as e:
        return f"Invalid date format. Please use YYYY-MM-DD. Error: {e}"

    # Acquire lock for the entire ChromaDB operation
    with chroma_ref_lock:
        if not chroma_collection:
            return "Vector store not initialized. Please restart the server."

        # Build where clause for metadata filtering
        where_conditions = []

        # Add date range filters if provided
        if since_timestamp and until_timestamp:
            where_conditions.append({'modified': {'$gte': since_timestamp}})
            where_conditions.append({'modified': {'$lte': until_timestamp}})
        elif since_timestamp:
            where_conditions.append({'modified': {'$gte': since_timestamp}})
        elif until_timestamp:
            where_conditions.append({'modified': {'$lte': until_timestamp}})

        # Add vault filter if specified
        if vault_filter:
            where_conditions.append({'vault': vault_filter})

        # Combine metadata conditions
        where_clause = None
        if len(where_conditions) > 1:
            where_clause = {'$and': where_conditions}
        elif where_conditions:
            where_clause = where_conditions[0]

        # Build where_document clause for phrase filtering
        doc_conditions = []

        for phrase in parsed['required_phrases']:
            doc_conditions.append({"$contains": phrase})

        for phrase in parsed['excluded_phrases']:
            doc_conditions.append({"$not_contains": phrase})

        # Combine document conditions
        where_document = None
        if len(doc_conditions) > 1:
            where_document = {"$and": doc_conditions}
        elif doc_conditions:
            where_document = doc_conditions[0]

        # Execute search based on what we have
        if parsed['semantic_query']:
            # Semantic search with optional phrase and temporal filters
            results = chroma_collection.query(
                query_texts=[parsed['semantic_query']],
                n_results=limit,
                where=where_clause,
                where_document=where_document
            )
        elif where_document or where_clause:
            # Pure phrase/temporal search (no semantic component)
            results = chroma_collection.get(
                limit=limit,
                where=where_clause,
                where_document=where_document,
                include=['documents', 'metadatas']
            )
            # Restructure to match query() format
            if results['documents']:
                results = {
                    'documents': [results['documents']],
                    'metadatas': [results['metadatas']],
                    'distances': [[0.5] * len(results['documents'])]  # Default distance
                }
            else:
                results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        else:
            # No query at all
            return "Please provide a search query."

    # Process and format the results
    formatted_results = []
    has_temporal_filter = since_date or until_date

    if results["documents"] and results["documents"][0]:
        # Collect all docs with metadata
        docs_with_metadata = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            docs_with_metadata.append((doc, metadata, distance))

        # Apply sorting based on sort_by parameter (default to relevance)
        if sort_by or sort_order != "desc":  # If sort parameters were explicitly set
            docs_with_metadata = _sort_results(docs_with_metadata, sort_by or "relevance", sort_order)
        elif not parsed['semantic_query']:  # No semantic query means no relevance sorting needed
            # Default to date sorting for non-semantic queries
            docs_with_metadata = _sort_results(docs_with_metadata, "date", "desc")
        # else: Keep ChromaDB's relevance order (already sorted by distance)

        # Format the sorted results
        for doc, metadata, distance in docs_with_metadata:
            # Show date if temporal filter is used
            formatted_results.append(_format_search_result(doc, metadata, distance, parsed, show_date=has_temporal_filter))

    if not formatted_results:
        error_parts = []
        if parsed['required_phrases']:
            error_parts.append(f"containing: {', '.join(['\"' + p + '\"' for p in parsed['required_phrases']])}")
        if since_date or until_date:
            date_parts = []
            if since_date:
                date_parts.append(f"since {since_date}")
            if until_date:
                date_parts.append(f"until {until_date}")
            error_parts.append(" and ".join(date_parts))

        if error_parts:
            return f"No results found {' '.join(error_parts)}."
        return "No results found for your query."

    # Return results as a single formatted string
    return "\n\n---\n\n".join(formatted_results)


def _sort_results(docs_with_metadata: List[Tuple], sort_by: str, sort_order: str) -> List[Tuple]:
    """
    Sort search results based on the specified criteria.

    Args:
        docs_with_metadata: List of tuples (doc, metadata, distance)
        sort_by: Sort criteria ("relevance", "date", "title", "path")
        sort_order: Sort order ("asc" or "desc")

    Returns:
        Sorted list of tuples
    """
    # Define sort key functions for each type
    if sort_by == "date":
        # Sort by modification timestamp
        sort_key = lambda x: x[1].get('modified', 0)
    elif sort_by == "title":
        # Sort alphabetically by title (case-insensitive)
        sort_key = lambda x: x[1].get('title', '').lower()
    elif sort_by == "path":
        # Sort by relative path (source)
        sort_key = lambda x: x[1].get('source', '').lower()
    else:  # Default to "relevance"
        # Sort by distance (lower is better)
        # Note: When sorting by relevance, we want LOWER distances first (better matches)
        # So for desc order (default), we actually use ascending sort on distance
        sort_key = lambda x: x[2] if x[2] is not None else float('inf')

    # Apply sort
    reverse = (sort_order == "desc")

    # Special case for relevance: desc means best matches first (lowest distance)
    if sort_by == "relevance" or sort_by is None:
        reverse = not reverse  # Invert for relevance

    return sorted(docs_with_metadata, key=sort_key, reverse=reverse)


def _format_search_result(doc: str, metadata: Dict, distance: Optional[float], parsed: Dict, show_date: bool = False) -> str:
    """Helper function to format a single search result"""
    # Create preview and highlight matched phrases
    preview = doc[:PREVIEW_LENGTH]
    if len(doc) > PREVIEW_LENGTH:
        preview += '...'

    # Bold the required phrases in preview
    for phrase in parsed.get('required_phrases', []):
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        preview = pattern.sub(lambda m: f"**{m.group()}**", preview)

    # Get full path
    full_path = get_full_path_from_metadata(metadata)

    # Build result text
    result_text = f"**{metadata.get('title', 'Untitled')}** ({metadata.get('vault', 'Unknown vault')})\n"
    result_text += f"Path: {full_path}\n"

    # Add file size if available
    file_size_bytes = metadata.get('file_size_bytes')
    if file_size_bytes:
        # Format file size for human readability
        if file_size_bytes < 1024:
            size_str = f"{file_size_bytes} bytes"
        elif file_size_bytes < 1024 * 1024:
            size_str = f"{file_size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{file_size_bytes / (1024 * 1024):.1f} MB"
        result_text += f"Size: {size_str}\n"

    # Add date info if requested
    if show_date:
        modified_timestamp = metadata.get('modified', 0)
        if modified_timestamp:
            modified_dt = datetime.fromtimestamp(modified_timestamp)
            modified_str = modified_dt.strftime("%Y-%m-%d %H:%M")

            # Calculate relative time
            now = datetime.now()
            time_diff = now - modified_dt
            if time_diff.days == 0:
                if time_diff.seconds < 3600:
                    relative_time = f"{time_diff.seconds // 60} minutes ago"
                else:
                    relative_time = f"{time_diff.seconds // 3600} hours ago"
            elif time_diff.days == 1:
                relative_time = "Yesterday"
            elif time_diff.days < 7:
                relative_time = f"{time_diff.days} days ago"
            elif time_diff.days < 30:
                relative_time = f"{time_diff.days // 7} weeks ago"
            else:
                relative_time = f"{time_diff.days // 30} months ago"

            result_text += f"Modified: {modified_str} ({relative_time})\n"

    result_text += f"Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}\n"

    if distance is not None:
        result_text += f"Score: {1 - distance if isinstance(distance, (int, float)) else 'N/A'}\n"

    result_text += f"\n{preview}"

    return result_text


@mcp.tool
async def reindex_vaults() -> str:
    """
    Manually trigger a re-index of all configured Obsidian vaults.

    This will:
    - Check all configured vaults for new, modified, or deleted files
    - Update the vector database with any changes
    - Return a summary of the indexing operation
    """
    global chroma_client, chroma_collection

    config = load_config()
    vaults = config.get("vaults", [])

    if not vaults:
        return "No vaults configured. Please configure vaults first using 'mcp-obsidian configure'."

    try:
        # Run re-indexing
        start_time = time.time()

        # Run the update in a thread to avoid blocking the async context
        loop = asyncio.get_event_loop()
        new_client, new_collection = await loop.run_in_executor(
            None, initialize_vector_store, vaults
        )

        # Update global references atomically
        with chroma_ref_lock:
            chroma_client = new_client
            chroma_collection = new_collection

        elapsed_time = time.time() - start_time

        # Get statistics
        total_docs = new_collection.count()

        return (
            f"✅ Manual re-indexing completed successfully!\n\n"
            f"📊 Statistics:\n"
            f"• Vaults indexed: {len(vaults)}\n"
            f"• Total documents in database: {total_docs}\n"
            f"• Time taken: {elapsed_time:.2f} seconds\n\n"
            f"The vector store has been updated with the latest changes from your Obsidian vaults."
        )

    except Exception as e:
        return f"❌ Re-indexing failed with error: {str(e)}\n\nPlease check your vault configurations and try again."


class VaultChangeHandler(FileSystemEventHandler):
    """Handle file system changes in Obsidian vaults"""

    def __init__(self, vaults: List[Dict], update_callback):
        self.vaults = vaults
        self.update_callback = update_callback
        self.last_update = 0
        self.pending_update = False
        self.debounce_seconds = MONITOR_DEBOUNCE_SECONDS

    def on_any_event(self, event: FileSystemEvent):
        """Handle any file system event"""
        # Only care about markdown files
        if not event.src_path.endswith('.md'):
            return

        # Skip hidden files/directories
        if any(part.startswith('.') for part in Path(event.src_path).parts):
            return

        # Mark that we have a pending update
        self.pending_update = True
        self.last_update = time.time()

        # Log the change
        event_type = event.event_type
        file_name = Path(event.src_path).name
        print(f"\n📝 Detected {event_type}: {file_name}", file=sys.stderr)


async def monitor_vaults(vaults: List[Dict], update_interval: int = MONITOR_UPDATE_INTERVAL):
    """Monitor vaults for changes and trigger re-indexing"""
    global chroma_client, chroma_collection

    # Create file system event handler
    handler = VaultChangeHandler(vaults, lambda: initialize_vector_store(vaults))

    # Set up observers for each vault
    observers = []
    for vault in vaults:
        vault_path = Path(vault["path"])
        if vault_path.exists():
            observer = Observer()
            observer.schedule(handler, str(vault_path), recursive=True)
            observer.start()
            observers.append(observer)
            print(f"👁️  Monitoring vault: {vault['name']}", file=sys.stderr)

    try:
        while True:
            await asyncio.sleep(update_interval)

            # Check if we have pending updates and enough time has passed
            if handler.pending_update:
                time_since_last = time.time() - handler.last_update
                if time_since_last >= handler.debounce_seconds:
                    print(f"\n🔄 Re-indexing vaults after changes...", file=sys.stderr)
                    handler.pending_update = False

                    # Run the update in a thread to avoid blocking
                    loop = asyncio.get_event_loop()
                    new_client, new_collection = await loop.run_in_executor(
                        None, initialize_vector_store, vaults
                    )

                    # Update global references atomically
                    with chroma_ref_lock:
                        chroma_client = new_client
                        chroma_collection = new_collection

                    print("✅ Re-indexing complete! Ready for queries.", file=sys.stderr)

    except asyncio.CancelledError:
        # Clean shutdown
        for observer in observers:
            observer.stop()
            observer.join()
        raise


def serve():
    """Run the MCP server."""
    global chroma_client, chroma_collection

    config = load_config()
    vaults = config.get("vaults", [])

    if not vaults:
        print("No vaults configured. Run with 'configure' first.", file=sys.stderr)
        sys.exit(1)

    print("🚀 Starting MCP Obsidian server...", file=sys.stderr)

    # Initialize vector store
    new_client, new_collection = initialize_vector_store(vaults)

    # Update global references atomically
    with chroma_ref_lock:
        chroma_client = new_client
        chroma_collection = new_collection

    print("\n📡 MCP server ready!", file=sys.stderr)
    print("Vector store initialized and ready for queries.", file=sys.stderr)

    # Start the file system monitor in a background thread
    # Since FastMCP doesn't support startup/shutdown hooks, we'll use a simpler approach
    import threading

    def start_monitor():
        """Start monitoring in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(monitor_vaults(vaults))
        except KeyboardInterrupt:
            pass

    # Start monitoring in a background thread
    monitor_thread = threading.Thread(target=start_monitor, daemon=True)
    monitor_thread.start()
    print("👁️  File system monitoring started", file=sys.stderr)

    # Run the FastMCP server (it handles its own event loop)
    mcp.run()


def index_vaults(force=False):
    """Rebuild search index for all configured vaults"""
    global chroma_client, chroma_collection

    config = load_config()
    vaults = config.get("vaults", [])

    if not vaults:
        print("No vaults configured. Run 'mcp-obsidian configure' first.", file=sys.stderr)
        sys.exit(1)

    print("🔄 Rebuilding search index for all configured vaults...", file=sys.stderr)
    print(f"Found {len(vaults)} vault(s) to index.\n", file=sys.stderr)

    # Initialize vector store with force flag (handles locking internally)
    new_client, new_collection = initialize_vector_store(vaults, force_rebuild=force)

    # Update global references atomically
    with chroma_ref_lock:
        chroma_client = new_client
        chroma_collection = new_collection

    # Show results
    if new_collection:
        doc_count = new_collection.count()
        print(f"\n✅ Index rebuilt successfully!", file=sys.stderr)
        print(f"📊 Total documents indexed: {doc_count}", file=sys.stderr)

        # Show vaults that were indexed
        print("\n📁 Vaults indexed:", file=sys.stderr)
        for vault in vaults:
            print(f"   - {vault['name']} ({vault['path']})", file=sys.stderr)
    else:
        print("❌ Failed to rebuild index.", file=sys.stderr)
        sys.exit(1)


def load_config():
    """Load configuration"""
    if not CONFIG_FILE.exists():
        return {"vaults": []}

    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def main():
    """Main entry point for the MCP Obsidian server."""
    parser = argparse.ArgumentParser(description="MCP Obsidian Server")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Configure subcommand
    subparsers.add_parser("configure", help="Configure vault paths")

    # Index subcommand
    index_parser = subparsers.add_parser("index", help="Rebuild search index for all configured vaults")
    index_parser.add_argument("-f", "--force", action="store_true",
                             help="Force complete rebuild by clearing existing index")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "configure":
        configure()
    elif args.command == "index":
        index_vaults(force=args.force)
    else:
        # Default to server mode
        serve()


if __name__ == "__main__":
    main()
