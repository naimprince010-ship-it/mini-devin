"""
LSP (Language Server Protocol) Module for Mini-Devin

This module provides Language Server Protocol integration for:
- Go-to-definition: Find where symbols are defined
- Find-references: Find all usages of a symbol
- Real-time diagnostics: Get errors, warnings, and hints
- Hover information: Get documentation and type info
- Code completion: Get suggestions while typing
"""

from .client import LSPClient, LSPClientConfig, get_language_id, LANGUAGE_SERVERS
from .manager import LSPManager, LSPManagerConfig, create_lsp_manager
from .types import (
    Position,
    Range,
    Location,
    Diagnostic,
    DiagnosticSeverity,
    CompletionItem,
    CompletionItemKind,
    Hover,
    SymbolInformation,
    SymbolKind,
    DocumentSymbol,
)
from .tools import (
    GotoDefinitionInput,
    GotoDefinitionOutput,
    FindReferencesInput,
    FindReferencesOutput,
    HoverInput,
    HoverOutput,
    GetCompletionsInput,
    GetCompletionsOutput,
    GetDiagnosticsInput,
    GetDiagnosticsOutput,
    GetDocumentSymbolsInput,
    GetDocumentSymbolsOutput,
    SearchWorkspaceSymbolsInput,
    SearchWorkspaceSymbolsOutput,
    LSP_TOOL_SCHEMAS,
)

__all__ = [
    # Client
    "LSPClient",
    "LSPClientConfig",
    "get_language_id",
    "LANGUAGE_SERVERS",
    # Manager
    "LSPManager",
    "LSPManagerConfig",
    "create_lsp_manager",
    # Types
    "Position",
    "Range",
    "Location",
    "Diagnostic",
    "DiagnosticSeverity",
    "CompletionItem",
    "CompletionItemKind",
    "Hover",
    "SymbolInformation",
    "SymbolKind",
    "DocumentSymbol",
    # Tool schemas
    "GotoDefinitionInput",
    "GotoDefinitionOutput",
    "FindReferencesInput",
    "FindReferencesOutput",
    "HoverInput",
    "HoverOutput",
    "GetCompletionsInput",
    "GetCompletionsOutput",
    "GetDiagnosticsInput",
    "GetDiagnosticsOutput",
    "GetDocumentSymbolsInput",
    "GetDocumentSymbolsOutput",
    "SearchWorkspaceSymbolsInput",
    "SearchWorkspaceSymbolsOutput",
    "LSP_TOOL_SCHEMAS",
]
