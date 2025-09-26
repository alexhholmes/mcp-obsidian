#!/usr/bin/env python3
import argparse
import json
import sys
import hashlib
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

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


CONFIG_DIR = Path.home() / ".mcp-obsidian"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Create FastMCP server instance
mcp = FastMCP("mcp-obsidian")

# Global ChromaDB client and collection
chroma_client: Optional[chromadb.Client] = None
chroma_collection: Optional[chromadb.Collection] = None

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
                    {"name": "Exit (^C)", "value": "4", "shortcut_key": "4"}
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
        elif choice == "4" or choice is None:
            break

        print(file=sys.stderr)  # Add spacing between operations


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
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

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
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
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


def initialize_vector_store(vaults: List[Dict]) -> Tuple[chromadb.Client, chromadb.Collection]:
    """Initialize ChromaDB and incrementally index vault content"""
    print("🔄 Initializing vector store...", file=sys.stderr)

    # Disable ChromaDB telemetry to avoid errors
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    # Initialize ChromaDB with persistent storage
    db_path = CONFIG_DIR / "chroma_db"
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
        name="obsidian_notes",
        metadata={"description": "Obsidian vault notes"}
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

                    # Add to collection
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


@mcp.tool
async def semantic_search(
    query: str,
    vault_filter: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Search Obsidian vault notes using semantic similarity.

    Args:
        query: The search query to find similar content
        vault_filter: Optional filter results to a specific vault name
        limit: Maximum number of results to return (default: 10, max: 20)
    """
    global chroma_collection

    if not chroma_collection:
        return "Vector store not initialized. Please restart the server."

    # Cap limit at 20
    limit = min(limit, 20)

    # Build where clause for filtering
    where_clause = None
    if vault_filter:
        where_clause = {"vault": vault_filter}

    # Query the collection
    results = chroma_collection.query(
        query_texts=[query],
        n_results=limit,
        where=where_clause
    )

    # Format the results
    formatted_results = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None

            result_text = (
                f"**{metadata.get('title', 'Untitled')}** ({metadata.get('vault', 'Unknown vault')})\n"
                f"Source: {metadata.get('source', 'Unknown')}\n"
                f"Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}\n"
                f"Score: {1 - distance if distance else 'N/A'}\n\n"
                f"{doc[:500]}{'...' if len(doc) > 500 else ''}"
            )
            formatted_results.append(result_text)

    if not formatted_results:
        return "No results found for your query."

    # Return results as a single formatted string
    return "\n\n---\n\n".join(formatted_results) if formatted_results else "No results found."


class VaultChangeHandler(FileSystemEventHandler):
    """Handle file system changes in Obsidian vaults"""

    def __init__(self, vaults: List[Dict], update_callback):
        self.vaults = vaults
        self.update_callback = update_callback
        self.last_update = 0
        self.pending_update = False
        self.debounce_seconds = 30  # Wait 30 seconds after last change before updating

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


async def monitor_vaults(vaults: List[Dict], update_interval: int = 10):
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

                    # Update global references
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
    chroma_client, chroma_collection = initialize_vector_store(vaults)

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

    # Parse arguments
    args = parser.parse_args()

    if args.command == "configure":
        configure()
    else:
        # Default to server mode
        serve()


if __name__ == "__main__":
    main()
