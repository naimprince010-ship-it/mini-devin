"""
LSP Client Implementation

This module provides a client for communicating with Language Servers
using the Language Server Protocol over stdio.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

from .types import (
    Position,
    Location,
    Diagnostic,
    CompletionItem,
    Hover,
    SymbolInformation,
    DocumentSymbol,
    ServerCapabilities,
    TextDocumentItem,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
    TextDocumentContentChangeEvent,
    WorkspaceFolder,
)


LANGUAGE_SERVERS: dict[str, dict[str, Any]] = {
    "python": {
        "command": ["pylsp"],
        "alt_commands": [
            ["pyright-langserver", "--stdio"],
            ["python", "-m", "pylsp"],
        ],
        "extensions": [".py", ".pyi"],
        "language_id": "python",
    },
    "typescript": {
        "command": ["typescript-language-server", "--stdio"],
        "alt_commands": [],
        "extensions": [".ts", ".tsx"],
        "language_id": "typescript",
    },
    "javascript": {
        "command": ["typescript-language-server", "--stdio"],
        "alt_commands": [],
        "extensions": [".js", ".jsx", ".mjs", ".cjs"],
        "language_id": "javascript",
    },
    "rust": {
        "command": ["rust-analyzer"],
        "alt_commands": [],
        "extensions": [".rs"],
        "language_id": "rust",
    },
    "go": {
        "command": ["gopls"],
        "alt_commands": [],
        "extensions": [".go"],
        "language_id": "go",
    },
    "java": {
        "command": ["jdtls"],
        "alt_commands": [],
        "extensions": [".java"],
        "language_id": "java",
    },
    "c": {
        "command": ["clangd"],
        "alt_commands": [],
        "extensions": [".c", ".h"],
        "language_id": "c",
    },
    "cpp": {
        "command": ["clangd"],
        "alt_commands": [],
        "extensions": [".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"],
        "language_id": "cpp",
    },
}


def get_language_id(file_path: str) -> str | None:
    """Get the language ID for a file based on its extension."""
    ext = Path(file_path).suffix.lower()
    for lang, config in LANGUAGE_SERVERS.items():
        if ext in config["extensions"]:
            return config["language_id"]
    return None


def get_server_command(language: str) -> list[str] | None:
    """Get the command to start a language server for a language."""
    if language in LANGUAGE_SERVERS:
        return LANGUAGE_SERVERS[language]["command"]
    return None


@dataclass
class LSPClientConfig:
    """Configuration for the LSP client."""
    language: str
    workspace_path: str
    server_command: list[str] | None = None
    initialization_options: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    auto_start: bool = True


class LSPClient:
    """
    Language Server Protocol client.
    
    Communicates with language servers over stdio using JSON-RPC 2.0.
    """
    
    def __init__(self, config: LSPClientConfig):
        self.config = config
        self.workspace_path = os.path.abspath(config.workspace_path)
        self.workspace_uri = f"file://{self.workspace_path}"
        
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future[Any]] = {}
        self._notification_handlers: dict[str, Callable[[dict[str, Any]], None]] = {}
        
        self._initialized = False
        self._capabilities: ServerCapabilities | None = None
        self._open_documents: dict[str, int] = {}  # uri -> version
        
        self._diagnostics: dict[str, list[Diagnostic]] = {}  # uri -> diagnostics
        self._read_task: asyncio.Task[None] | None = None
    
    @property
    def is_running(self) -> bool:
        """Check if the language server is running."""
        return self._process is not None and self._process.returncode is None
    
    @property
    def capabilities(self) -> ServerCapabilities | None:
        """Get the server capabilities."""
        return self._capabilities
    
    def _get_server_command(self) -> list[str]:
        """Get the command to start the language server."""
        if self.config.server_command:
            return self.config.server_command
        
        cmd = get_server_command(self.config.language)
        if cmd:
            return cmd
        
        raise ValueError(f"No language server configured for: {self.config.language}")
    
    async def start(self) -> bool:
        """Start the language server."""
        if self.is_running:
            return True
        
        try:
            cmd = self._get_server_command()
            
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_path,
            )
            
            if self._process.stdout is None or self._process.stdin is None:
                return False
            
            self._reader = self._process.stdout
            self._writer = self._process.stdin
            
            self._read_task = asyncio.create_task(self._read_messages())
            
            success = await self._initialize()
            if success:
                self._initialized = True
            
            return success
            
        except FileNotFoundError:
            print(f"Language server not found: {self._get_server_command()}")
            return False
        except Exception as e:
            print(f"Error starting language server: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the language server."""
        if not self.is_running:
            return
        
        try:
            await self._send_request("shutdown", {})
            await self._send_notification("exit", {})
        except Exception:
            pass
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
        
        self._process = None
        self._reader = None
        self._writer = None
        self._initialized = False
        self._open_documents.clear()
        self._diagnostics.clear()
    
    async def _initialize(self) -> bool:
        """Initialize the language server."""
        params = {
            "processId": os.getpid(),
            "rootUri": self.workspace_uri,
            "rootPath": self.workspace_path,
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": True,
                        "willSave": True,
                        "willSaveWaitUntil": True,
                        "didSave": True,
                    },
                    "completion": {
                        "dynamicRegistration": True,
                        "completionItem": {
                            "snippetSupport": True,
                            "commitCharactersSupport": True,
                            "documentationFormat": ["markdown", "plaintext"],
                            "deprecatedSupport": True,
                        },
                    },
                    "hover": {
                        "dynamicRegistration": True,
                        "contentFormat": ["markdown", "plaintext"],
                    },
                    "definition": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                    "publishDiagnostics": {
                        "relatedInformation": True,
                        "tagSupport": {"valueSet": [1, 2]},
                    },
                },
                "workspace": {
                    "workspaceFolders": True,
                    "didChangeConfiguration": {"dynamicRegistration": True},
                },
            },
            "workspaceFolders": [
                WorkspaceFolder(
                    uri=self.workspace_uri,
                    name=os.path.basename(self.workspace_path),
                ).to_dict()
            ],
            "initializationOptions": self.config.initialization_options,
        }
        
        try:
            result = await self._send_request("initialize", params)
            if result and "capabilities" in result:
                self._capabilities = ServerCapabilities.from_dict(result["capabilities"])
                await self._send_notification("initialized", {})
                return True
            return False
        except Exception as e:
            print(f"Error initializing language server: {e}")
            return False
    
    def _next_request_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Send a request to the language server and wait for response."""
        if not self._writer:
            raise RuntimeError("Language server not running")
        
        request_id = self._next_request_id()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future
        
        await self._send_message(message)
        
        try:
            timeout = timeout or self.config.timeout_seconds
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            del self._pending_requests[request_id]
            raise TimeoutError(f"Request {method} timed out")
    
    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification to the language server (no response expected)."""
        if not self._writer:
            raise RuntimeError("Language server not running")
        
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        
        await self._send_message(message)
    
    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the language server."""
        if not self._writer:
            return
        
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        
        self._writer.write(header.encode() + content.encode())
        await self._writer.drain()
    
    async def _read_messages(self) -> None:
        """Read messages from the language server."""
        if not self._reader:
            return
        
        while True:
            try:
                headers: dict[str, str] = {}
                while True:
                    line = await self._reader.readline()
                    if not line:
                        return
                    
                    line_str = line.decode().strip()
                    if not line_str:
                        break
                    
                    if ":" in line_str:
                        key, value = line_str.split(":", 1)
                        headers[key.strip().lower()] = value.strip()
                
                content_length = int(headers.get("content-length", 0))
                if content_length == 0:
                    continue
                
                content = await self._reader.read(content_length)
                message = json.loads(content.decode())
                
                await self._handle_message(message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error reading message: {e}")
                break
    
    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle a message from the language server."""
        if "id" in message:
            if "method" in message:
                pass
            else:
                request_id = message["id"]
                if request_id in self._pending_requests:
                    future = self._pending_requests.pop(request_id)
                    if "error" in message:
                        future.set_exception(
                            RuntimeError(message["error"].get("message", "Unknown error"))
                        )
                    else:
                        future.set_result(message.get("result"))
        else:
            method = message.get("method", "")
            params = message.get("params", {})
            
            if method == "textDocument/publishDiagnostics":
                self._handle_diagnostics(params)
            elif method in self._notification_handlers:
                self._notification_handlers[method](params)
    
    def _handle_diagnostics(self, params: dict[str, Any]) -> None:
        """Handle diagnostics notification."""
        uri = params.get("uri", "")
        diagnostics_data = params.get("diagnostics", [])
        
        self._diagnostics[uri] = [
            Diagnostic.from_dict(d) for d in diagnostics_data
        ]
    
    def on_notification(self, method: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Register a handler for a notification method."""
        self._notification_handlers[method] = handler
    
    def _file_uri(self, file_path: str) -> str:
        """Convert a file path to a URI."""
        abs_path = os.path.abspath(file_path)
        return f"file://{abs_path}"
    
    async def open_document(self, file_path: str, text: str | None = None) -> bool:
        """Open a document in the language server."""
        if not self._initialized:
            return False
        
        uri = self._file_uri(file_path)
        
        if uri in self._open_documents:
            return True
        
        if text is None:
            try:
                with open(file_path, "r") as f:
                    text = f.read()
            except Exception:
                return False
        
        language_id = get_language_id(file_path) or self.config.language
        
        doc = TextDocumentItem(
            uri=uri,
            language_id=language_id,
            version=1,
            text=text,
        )
        
        await self._send_notification(
            "textDocument/didOpen",
            {"textDocument": doc.to_dict()},
        )
        
        self._open_documents[uri] = 1
        return True
    
    async def close_document(self, file_path: str) -> None:
        """Close a document in the language server."""
        if not self._initialized:
            return
        
        uri = self._file_uri(file_path)
        
        if uri not in self._open_documents:
            return
        
        await self._send_notification(
            "textDocument/didClose",
            {"textDocument": TextDocumentIdentifier(uri=uri).to_dict()},
        )
        
        del self._open_documents[uri]
        self._diagnostics.pop(uri, None)
    
    async def update_document(
        self,
        file_path: str,
        changes: list[TextDocumentContentChangeEvent] | str,
    ) -> None:
        """Update a document in the language server."""
        if not self._initialized:
            return
        
        uri = self._file_uri(file_path)
        
        if uri not in self._open_documents:
            if isinstance(changes, str):
                await self.open_document(file_path, changes)
            return
        
        version = self._open_documents[uri] + 1
        self._open_documents[uri] = version
        
        if isinstance(changes, str):
            content_changes = [{"text": changes}]
        else:
            content_changes = [c.to_dict() for c in changes]
        
        await self._send_notification(
            "textDocument/didChange",
            {
                "textDocument": VersionedTextDocumentIdentifier(
                    uri=uri, version=version
                ).to_dict(),
                "contentChanges": content_changes,
            },
        )
    
    async def goto_definition(
        self,
        file_path: str,
        position: Position,
    ) -> list[Location]:
        """Go to the definition of a symbol."""
        if not self._initialized:
            return []
        
        if self._capabilities and not self._capabilities.definition_provider:
            return []
        
        uri = self._file_uri(file_path)
        
        if uri not in self._open_documents:
            await self.open_document(file_path)
        
        params = {
            "textDocument": TextDocumentIdentifier(uri=uri).to_dict(),
            "position": position.to_dict(),
        }
        
        try:
            result = await self._send_request("textDocument/definition", params)
            
            if result is None:
                return []
            
            if isinstance(result, dict):
                return [Location.from_dict(result)]
            elif isinstance(result, list):
                return [Location.from_dict(loc) for loc in result]
            
            return []
        except Exception as e:
            print(f"Error getting definition: {e}")
            return []
    
    async def find_references(
        self,
        file_path: str,
        position: Position,
        include_declaration: bool = True,
    ) -> list[Location]:
        """Find all references to a symbol."""
        if not self._initialized:
            return []
        
        if self._capabilities and not self._capabilities.references_provider:
            return []
        
        uri = self._file_uri(file_path)
        
        if uri not in self._open_documents:
            await self.open_document(file_path)
        
        params = {
            "textDocument": TextDocumentIdentifier(uri=uri).to_dict(),
            "position": position.to_dict(),
            "context": {"includeDeclaration": include_declaration},
        }
        
        try:
            result = await self._send_request("textDocument/references", params)
            
            if result is None:
                return []
            
            return [Location.from_dict(loc) for loc in result]
        except Exception as e:
            print(f"Error finding references: {e}")
            return []
    
    async def hover(self, file_path: str, position: Position) -> Hover | None:
        """Get hover information for a position."""
        if not self._initialized:
            return None
        
        if self._capabilities and not self._capabilities.hover_provider:
            return None
        
        uri = self._file_uri(file_path)
        
        if uri not in self._open_documents:
            await self.open_document(file_path)
        
        params = {
            "textDocument": TextDocumentIdentifier(uri=uri).to_dict(),
            "position": position.to_dict(),
        }
        
        try:
            result = await self._send_request("textDocument/hover", params)
            
            if result is None:
                return None
            
            return Hover.from_dict(result)
        except Exception as e:
            print(f"Error getting hover: {e}")
            return None
    
    async def get_completions(
        self,
        file_path: str,
        position: Position,
    ) -> list[CompletionItem]:
        """Get completion items at a position."""
        if not self._initialized:
            return []
        
        if self._capabilities and not self._capabilities.completion_provider:
            return []
        
        uri = self._file_uri(file_path)
        
        if uri not in self._open_documents:
            await self.open_document(file_path)
        
        params = {
            "textDocument": TextDocumentIdentifier(uri=uri).to_dict(),
            "position": position.to_dict(),
        }
        
        try:
            result = await self._send_request("textDocument/completion", params)
            
            if result is None:
                return []
            
            items = result.get("items", result) if isinstance(result, dict) else result
            return [CompletionItem.from_dict(item) for item in items]
        except Exception as e:
            print(f"Error getting completions: {e}")
            return []
    
    async def get_document_symbols(
        self,
        file_path: str,
    ) -> list[SymbolInformation | DocumentSymbol]:
        """Get symbols in a document."""
        if not self._initialized:
            return []
        
        if self._capabilities and not self._capabilities.document_symbol_provider:
            return []
        
        uri = self._file_uri(file_path)
        
        if uri not in self._open_documents:
            await self.open_document(file_path)
        
        params = {
            "textDocument": TextDocumentIdentifier(uri=uri).to_dict(),
        }
        
        try:
            result = await self._send_request("textDocument/documentSymbol", params)
            
            if result is None:
                return []
            
            symbols: list[SymbolInformation | DocumentSymbol] = []
            for item in result:
                if "location" in item:
                    symbols.append(SymbolInformation.from_dict(item))
                else:
                    symbols.append(DocumentSymbol.from_dict(item))
            
            return symbols
        except Exception as e:
            print(f"Error getting document symbols: {e}")
            return []
    
    async def get_workspace_symbols(self, query: str = "") -> list[SymbolInformation]:
        """Search for symbols in the workspace."""
        if not self._initialized:
            return []
        
        if self._capabilities and not self._capabilities.workspace_symbol_provider:
            return []
        
        params = {"query": query}
        
        try:
            result = await self._send_request("workspace/symbol", params)
            
            if result is None:
                return []
            
            return [SymbolInformation.from_dict(item) for item in result]
        except Exception as e:
            print(f"Error getting workspace symbols: {e}")
            return []
    
    def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Get diagnostics for a file."""
        uri = self._file_uri(file_path)
        return self._diagnostics.get(uri, [])
    
    def get_all_diagnostics(self) -> dict[str, list[Diagnostic]]:
        """Get all diagnostics."""
        return dict(self._diagnostics)
    
    async def __aenter__(self) -> "LSPClient":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
