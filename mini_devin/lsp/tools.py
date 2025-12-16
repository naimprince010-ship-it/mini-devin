"""
LSP Tool Schemas

This module provides tool schemas for LSP operations that can be used
by the agent to interact with language servers.
"""

from dataclasses import dataclass, field
from typing import Any

from .types import (
    Location,
    Diagnostic,
    DiagnosticSeverity,
    Hover,
    CompletionItem,
    SymbolInformation,
    DocumentSymbol,
    SymbolKind,
)


@dataclass
class GotoDefinitionInput:
    """Input for go-to-definition tool."""
    file_path: str
    line: int  # 1-indexed for user convenience
    character: int  # 1-indexed for user convenience
    
    def to_zero_indexed(self) -> tuple[int, int]:
        """Convert to 0-indexed line and character."""
        return (self.line - 1, self.character - 1)


@dataclass
class GotoDefinitionOutput:
    """Output for go-to-definition tool."""
    success: bool
    locations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    
    @classmethod
    def from_locations(cls, locations: list[Location]) -> "GotoDefinitionOutput":
        return cls(
            success=len(locations) > 0,
            locations=[
                {
                    "file_path": loc.file_path,
                    "line": loc.range.start.line + 1,
                    "character": loc.range.start.character + 1,
                    "end_line": loc.range.end.line + 1,
                    "end_character": loc.range.end.character + 1,
                }
                for loc in locations
            ],
        )
    
    @classmethod
    def from_error(cls, error: str) -> "GotoDefinitionOutput":
        return cls(success=False, error=error)


@dataclass
class FindReferencesInput:
    """Input for find-references tool."""
    file_path: str
    line: int  # 1-indexed
    character: int  # 1-indexed
    include_declaration: bool = True
    
    def to_zero_indexed(self) -> tuple[int, int]:
        """Convert to 0-indexed line and character."""
        return (self.line - 1, self.character - 1)


@dataclass
class FindReferencesOutput:
    """Output for find-references tool."""
    success: bool
    references: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0
    error: str | None = None
    
    @classmethod
    def from_locations(cls, locations: list[Location]) -> "FindReferencesOutput":
        return cls(
            success=True,
            references=[
                {
                    "file_path": loc.file_path,
                    "line": loc.range.start.line + 1,
                    "character": loc.range.start.character + 1,
                    "end_line": loc.range.end.line + 1,
                    "end_character": loc.range.end.character + 1,
                }
                for loc in locations
            ],
            count=len(locations),
        )
    
    @classmethod
    def from_error(cls, error: str) -> "FindReferencesOutput":
        return cls(success=False, error=error)


@dataclass
class HoverInput:
    """Input for hover tool."""
    file_path: str
    line: int  # 1-indexed
    character: int  # 1-indexed
    
    def to_zero_indexed(self) -> tuple[int, int]:
        """Convert to 0-indexed line and character."""
        return (self.line - 1, self.character - 1)


@dataclass
class HoverOutput:
    """Output for hover tool."""
    success: bool
    contents: str | None = None
    error: str | None = None
    
    @classmethod
    def from_hover(cls, hover: Hover | None) -> "HoverOutput":
        if hover is None:
            return cls(success=False, error="No hover information available")
        return cls(success=True, contents=hover.contents)
    
    @classmethod
    def from_error(cls, error: str) -> "HoverOutput":
        return cls(success=False, error=error)


@dataclass
class GetCompletionsInput:
    """Input for get-completions tool."""
    file_path: str
    line: int  # 1-indexed
    character: int  # 1-indexed
    max_items: int = 20
    
    def to_zero_indexed(self) -> tuple[int, int]:
        """Convert to 0-indexed line and character."""
        return (self.line - 1, self.character - 1)


@dataclass
class GetCompletionsOutput:
    """Output for get-completions tool."""
    success: bool
    completions: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0
    error: str | None = None
    
    @classmethod
    def from_items(
        cls,
        items: list[CompletionItem],
        max_items: int = 20,
    ) -> "GetCompletionsOutput":
        completions = [
            {
                "label": item.label,
                "kind": item.kind.name if item.kind else None,
                "detail": item.detail,
                "documentation": item.documentation,
                "insert_text": item.insert_text or item.label,
            }
            for item in items[:max_items]
        ]
        return cls(
            success=True,
            completions=completions,
            count=len(items),
        )
    
    @classmethod
    def from_error(cls, error: str) -> "GetCompletionsOutput":
        return cls(success=False, error=error)


@dataclass
class GetDiagnosticsInput:
    """Input for get-diagnostics tool."""
    file_path: str | None = None  # None means all files
    severity_filter: str | None = None  # "error", "warning", "info", "hint"


@dataclass
class GetDiagnosticsOutput:
    """Output for get-diagnostics tool."""
    success: bool
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    hint_count: int = 0
    error: str | None = None
    
    @classmethod
    def from_diagnostics(
        cls,
        diagnostics: list[tuple[str, Diagnostic]],
        severity_filter: str | None = None,
    ) -> "GetDiagnosticsOutput":
        severity_map = {
            "error": DiagnosticSeverity.ERROR,
            "warning": DiagnosticSeverity.WARNING,
            "info": DiagnosticSeverity.INFORMATION,
            "hint": DiagnosticSeverity.HINT,
        }
        
        filtered = diagnostics
        if severity_filter and severity_filter.lower() in severity_map:
            target_severity = severity_map[severity_filter.lower()]
            filtered = [(f, d) for f, d in diagnostics if d.severity == target_severity]
        
        error_count = sum(1 for _, d in diagnostics if d.severity == DiagnosticSeverity.ERROR)
        warning_count = sum(1 for _, d in diagnostics if d.severity == DiagnosticSeverity.WARNING)
        info_count = sum(1 for _, d in diagnostics if d.severity == DiagnosticSeverity.INFORMATION)
        hint_count = sum(1 for _, d in diagnostics if d.severity == DiagnosticSeverity.HINT)
        
        return cls(
            success=True,
            diagnostics=[
                {
                    "file_path": file_path,
                    "line": diag.range.start.line + 1,
                    "character": diag.range.start.character + 1,
                    "end_line": diag.range.end.line + 1,
                    "end_character": diag.range.end.character + 1,
                    "severity": diag.severity.name.lower(),
                    "message": diag.message,
                    "code": diag.code,
                    "source": diag.source,
                }
                for file_path, diag in filtered
            ],
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            hint_count=hint_count,
        )
    
    @classmethod
    def from_error(cls, error: str) -> "GetDiagnosticsOutput":
        return cls(success=False, error=error)


@dataclass
class GetDocumentSymbolsInput:
    """Input for get-document-symbols tool."""
    file_path: str
    kind_filter: str | None = None  # "function", "class", "method", etc.


@dataclass
class GetDocumentSymbolsOutput:
    """Output for get-document-symbols tool."""
    success: bool
    symbols: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0
    error: str | None = None
    
    @classmethod
    def from_symbols(
        cls,
        symbols: list[SymbolInformation | DocumentSymbol],
        kind_filter: str | None = None,
    ) -> "GetDocumentSymbolsOutput":
        kind_map = {
            "file": SymbolKind.FILE,
            "module": SymbolKind.MODULE,
            "namespace": SymbolKind.NAMESPACE,
            "package": SymbolKind.PACKAGE,
            "class": SymbolKind.CLASS,
            "method": SymbolKind.METHOD,
            "property": SymbolKind.PROPERTY,
            "field": SymbolKind.FIELD,
            "constructor": SymbolKind.CONSTRUCTOR,
            "enum": SymbolKind.ENUM,
            "interface": SymbolKind.INTERFACE,
            "function": SymbolKind.FUNCTION,
            "variable": SymbolKind.VARIABLE,
            "constant": SymbolKind.CONSTANT,
        }
        
        def symbol_to_dict(sym: SymbolInformation | DocumentSymbol) -> dict[str, Any]:
            if isinstance(sym, SymbolInformation):
                return {
                    "name": sym.name,
                    "kind": sym.kind.name.lower(),
                    "file_path": sym.location.file_path,
                    "line": sym.location.range.start.line + 1,
                    "character": sym.location.range.start.character + 1,
                    "container": sym.container_name,
                }
            else:
                return {
                    "name": sym.name,
                    "kind": sym.kind.name.lower(),
                    "detail": sym.detail,
                    "line": sym.range.start.line + 1,
                    "character": sym.range.start.character + 1,
                    "children": [symbol_to_dict(c) for c in sym.children] if sym.children else [],
                }
        
        result = [symbol_to_dict(s) for s in symbols]
        
        if kind_filter and kind_filter.lower() in kind_map:
            target_kind = kind_map[kind_filter.lower()].name.lower()
            result = [s for s in result if s["kind"] == target_kind]
        
        return cls(
            success=True,
            symbols=result,
            count=len(result),
        )
    
    @classmethod
    def from_error(cls, error: str) -> "GetDocumentSymbolsOutput":
        return cls(success=False, error=error)


@dataclass
class SearchWorkspaceSymbolsInput:
    """Input for search-workspace-symbols tool."""
    query: str
    language: str | None = None
    max_results: int = 50


@dataclass
class SearchWorkspaceSymbolsOutput:
    """Output for search-workspace-symbols tool."""
    success: bool
    symbols: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0
    error: str | None = None
    
    @classmethod
    def from_symbols(
        cls,
        symbols: list[SymbolInformation],
        max_results: int = 50,
    ) -> "SearchWorkspaceSymbolsOutput":
        return cls(
            success=True,
            symbols=[
                {
                    "name": sym.name,
                    "kind": sym.kind.name.lower(),
                    "file_path": sym.location.file_path,
                    "line": sym.location.range.start.line + 1,
                    "character": sym.location.range.start.character + 1,
                    "container": sym.container_name,
                }
                for sym in symbols[:max_results]
            ],
            count=len(symbols),
        )
    
    @classmethod
    def from_error(cls, error: str) -> "SearchWorkspaceSymbolsOutput":
        return cls(success=False, error=error)


LSP_TOOL_SCHEMAS = {
    "lsp_goto_definition": {
        "name": "lsp_goto_definition",
        "description": "Go to the definition of a symbol at a specific position in a file. Returns the location(s) where the symbol is defined.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file containing the symbol",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number (1-indexed)",
                },
                "character": {
                    "type": "integer",
                    "description": "Character position in the line (1-indexed)",
                },
            },
            "required": ["file_path", "line", "character"],
        },
    },
    "lsp_find_references": {
        "name": "lsp_find_references",
        "description": "Find all references to a symbol at a specific position. Returns all locations where the symbol is used.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file containing the symbol",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number (1-indexed)",
                },
                "character": {
                    "type": "integer",
                    "description": "Character position in the line (1-indexed)",
                },
                "include_declaration": {
                    "type": "boolean",
                    "description": "Whether to include the declaration in results (default: true)",
                    "default": True,
                },
            },
            "required": ["file_path", "line", "character"],
        },
    },
    "lsp_hover": {
        "name": "lsp_hover",
        "description": "Get hover information (documentation, type info) for a symbol at a specific position.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number (1-indexed)",
                },
                "character": {
                    "type": "integer",
                    "description": "Character position in the line (1-indexed)",
                },
            },
            "required": ["file_path", "line", "character"],
        },
    },
    "lsp_get_completions": {
        "name": "lsp_get_completions",
        "description": "Get code completion suggestions at a specific position.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number (1-indexed)",
                },
                "character": {
                    "type": "integer",
                    "description": "Character position in the line (1-indexed)",
                },
                "max_items": {
                    "type": "integer",
                    "description": "Maximum number of completions to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["file_path", "line", "character"],
        },
    },
    "lsp_get_diagnostics": {
        "name": "lsp_get_diagnostics",
        "description": "Get diagnostics (errors, warnings, hints) for a file or the entire workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (optional, omit for all files)",
                },
                "severity_filter": {
                    "type": "string",
                    "enum": ["error", "warning", "info", "hint"],
                    "description": "Filter by severity level (optional)",
                },
            },
            "required": [],
        },
    },
    "lsp_get_document_symbols": {
        "name": "lsp_get_document_symbols",
        "description": "Get all symbols (functions, classes, variables, etc.) defined in a document.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file",
                },
                "kind_filter": {
                    "type": "string",
                    "enum": ["function", "class", "method", "variable", "constant", "interface", "enum"],
                    "description": "Filter by symbol kind (optional)",
                },
            },
            "required": ["file_path"],
        },
    },
    "lsp_search_workspace_symbols": {
        "name": "lsp_search_workspace_symbols",
        "description": "Search for symbols across the entire workspace by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (symbol name or partial name)",
                },
                "language": {
                    "type": "string",
                    "description": "Filter by language (optional)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 50)",
                    "default": 50,
                },
            },
            "required": ["query"],
        },
    },
}
