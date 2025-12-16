"""
LSP Manager

This module provides a manager for handling multiple language servers
and coordinating LSP operations across different file types.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from .client import LSPClient, LSPClientConfig, get_language_id, LANGUAGE_SERVERS
from .types import (
    Position,
    Location,
    Diagnostic,
    CompletionItem,
    Hover,
    SymbolInformation,
    DocumentSymbol,
)


@dataclass
class LSPManagerConfig:
    """Configuration for the LSP manager."""
    workspace_path: str
    auto_start_servers: bool = True
    languages: list[str] = field(default_factory=list)
    timeout_seconds: int = 30


class LSPManager:
    """
    Manager for multiple language servers.
    
    Automatically starts and manages language servers based on file types
    in the workspace.
    """
    
    def __init__(self, config: LSPManagerConfig):
        self.config = config
        self.workspace_path = os.path.abspath(config.workspace_path)
        
        self._clients: dict[str, LSPClient] = {}
        self._starting: set[str] = set()
        self._failed: set[str] = set()
    
    @property
    def active_languages(self) -> list[str]:
        """Get list of languages with active servers."""
        return [lang for lang, client in self._clients.items() if client.is_running]
    
    def _get_client_for_file(self, file_path: str) -> LSPClient | None:
        """Get the appropriate LSP client for a file."""
        language = get_language_id(file_path)
        if language and language in self._clients:
            client = self._clients[language]
            if client.is_running:
                return client
        return None
    
    async def start(self, languages: list[str] | None = None) -> dict[str, bool]:
        """
        Start language servers.
        
        Args:
            languages: List of languages to start servers for.
                      If None, uses config.languages or auto-detects.
        
        Returns:
            Dict mapping language to success status.
        """
        if languages is None:
            languages = self.config.languages or self._detect_languages()
        
        results: dict[str, bool] = {}
        
        for language in languages:
            if language in self._clients and self._clients[language].is_running:
                results[language] = True
                continue
            
            if language in self._failed:
                results[language] = False
                continue
            
            if language not in LANGUAGE_SERVERS:
                results[language] = False
                continue
            
            success = await self._start_server(language)
            results[language] = success
        
        return results
    
    async def _start_server(self, language: str) -> bool:
        """Start a language server for a specific language."""
        if language in self._starting:
            return False
        
        self._starting.add(language)
        
        try:
            config = LSPClientConfig(
                language=language,
                workspace_path=self.workspace_path,
                timeout_seconds=self.config.timeout_seconds,
            )
            
            client = LSPClient(config)
            success = await client.start()
            
            if success:
                self._clients[language] = client
                return True
            else:
                self._failed.add(language)
                return False
                
        except Exception as e:
            print(f"Error starting {language} server: {e}")
            self._failed.add(language)
            return False
        finally:
            self._starting.discard(language)
    
    async def stop(self, languages: list[str] | None = None) -> None:
        """
        Stop language servers.
        
        Args:
            languages: List of languages to stop. If None, stops all.
        """
        if languages is None:
            languages = list(self._clients.keys())
        
        for language in languages:
            if language in self._clients:
                await self._clients[language].stop()
                del self._clients[language]
    
    def _detect_languages(self) -> list[str]:
        """Detect languages used in the workspace."""
        languages: set[str] = set()
        
        for root, _, files in os.walk(self.workspace_path):
            if ".git" in root or "node_modules" in root or "__pycache__" in root:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                lang = get_language_id(file_path)
                if lang:
                    languages.add(lang)
        
        return list(languages)
    
    async def ensure_server_for_file(self, file_path: str) -> LSPClient | None:
        """Ensure a language server is running for a file type."""
        language = get_language_id(file_path)
        if not language:
            return None
        
        if language in self._clients and self._clients[language].is_running:
            return self._clients[language]
        
        if language in self._failed:
            return None
        
        success = await self._start_server(language)
        if success:
            return self._clients.get(language)
        
        return None
    
    async def open_document(self, file_path: str, text: str | None = None) -> bool:
        """Open a document in the appropriate language server."""
        client = await self.ensure_server_for_file(file_path)
        if client:
            return await client.open_document(file_path, text)
        return False
    
    async def close_document(self, file_path: str) -> None:
        """Close a document in the appropriate language server."""
        client = self._get_client_for_file(file_path)
        if client:
            await client.close_document(file_path)
    
    async def update_document(self, file_path: str, text: str) -> None:
        """Update a document in the appropriate language server."""
        client = self._get_client_for_file(file_path)
        if client:
            await client.update_document(file_path, text)
    
    async def goto_definition(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> list[Location]:
        """
        Go to the definition of a symbol.
        
        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
        
        Returns:
            List of definition locations
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []
        
        position = Position(line=line, character=character)
        return await client.goto_definition(file_path, position)
    
    async def find_references(
        self,
        file_path: str,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> list[Location]:
        """
        Find all references to a symbol.
        
        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
            include_declaration: Whether to include the declaration
        
        Returns:
            List of reference locations
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []
        
        position = Position(line=line, character=character)
        return await client.find_references(file_path, position, include_declaration)
    
    async def hover(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> Hover | None:
        """
        Get hover information for a position.
        
        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
        
        Returns:
            Hover information or None
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return None
        
        position = Position(line=line, character=character)
        return await client.hover(file_path, position)
    
    async def get_completions(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> list[CompletionItem]:
        """
        Get completion items at a position.
        
        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
        
        Returns:
            List of completion items
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []
        
        position = Position(line=line, character=character)
        return await client.get_completions(file_path, position)
    
    async def get_document_symbols(
        self,
        file_path: str,
    ) -> list[SymbolInformation | DocumentSymbol]:
        """
        Get symbols in a document.
        
        Args:
            file_path: Path to the file
        
        Returns:
            List of symbols
        """
        client = await self.ensure_server_for_file(file_path)
        if not client:
            return []
        
        return await client.get_document_symbols(file_path)
    
    async def get_workspace_symbols(
        self,
        query: str = "",
        language: str | None = None,
    ) -> list[SymbolInformation]:
        """
        Search for symbols in the workspace.
        
        Args:
            query: Search query
            language: Specific language to search in (or all if None)
        
        Returns:
            List of matching symbols
        """
        results: list[SymbolInformation] = []
        
        clients = (
            [self._clients[language]] if language and language in self._clients
            else list(self._clients.values())
        )
        
        for client in clients:
            if client.is_running:
                symbols = await client.get_workspace_symbols(query)
                results.extend(symbols)
        
        return results
    
    def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """
        Get diagnostics for a file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            List of diagnostics
        """
        client = self._get_client_for_file(file_path)
        if client:
            return client.get_diagnostics(file_path)
        return []
    
    def get_all_diagnostics(self) -> dict[str, list[Diagnostic]]:
        """
        Get all diagnostics from all language servers.
        
        Returns:
            Dict mapping file URIs to diagnostics
        """
        all_diagnostics: dict[str, list[Diagnostic]] = {}
        
        for client in self._clients.values():
            if client.is_running:
                all_diagnostics.update(client.get_all_diagnostics())
        
        return all_diagnostics
    
    def get_errors(self, file_path: str | None = None) -> list[tuple[str, Diagnostic]]:
        """
        Get all error diagnostics.
        
        Args:
            file_path: Optional file to filter by
        
        Returns:
            List of (file_path, diagnostic) tuples for errors only
        """
        from .types import DiagnosticSeverity
        
        errors: list[tuple[str, Diagnostic]] = []
        
        if file_path:
            diagnostics = self.get_diagnostics(file_path)
            for diag in diagnostics:
                if diag.severity == DiagnosticSeverity.ERROR:
                    errors.append((file_path, diag))
        else:
            all_diags = self.get_all_diagnostics()
            for uri, diagnostics in all_diags.items():
                path = uri[7:] if uri.startswith("file://") else uri
                for diag in diagnostics:
                    if diag.severity == DiagnosticSeverity.ERROR:
                        errors.append((path, diag))
        
        return errors
    
    async def __aenter__(self) -> "LSPManager":
        """Async context manager entry."""
        if self.config.auto_start_servers:
            await self.start()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()


async def create_lsp_manager(
    workspace_path: str,
    languages: list[str] | None = None,
    auto_start: bool = True,
) -> LSPManager:
    """
    Create and optionally start an LSP manager.
    
    Args:
        workspace_path: Path to the workspace
        languages: Languages to start servers for (auto-detect if None)
        auto_start: Whether to start servers immediately
    
    Returns:
        Configured LSPManager instance
    """
    config = LSPManagerConfig(
        workspace_path=workspace_path,
        auto_start_servers=False,
        languages=languages or [],
    )
    
    manager = LSPManager(config)
    
    if auto_start:
        await manager.start(languages)
    
    return manager
