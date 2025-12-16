"""
LSP Type Definitions

This module defines the core types used in the Language Server Protocol.
Based on the LSP 3.17 specification.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class DiagnosticSeverity(IntEnum):
    """Severity of a diagnostic."""
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


class SymbolKind(IntEnum):
    """Kind of a symbol."""
    FILE = 1
    MODULE = 2
    NAMESPACE = 3
    PACKAGE = 4
    CLASS = 5
    METHOD = 6
    PROPERTY = 7
    FIELD = 8
    CONSTRUCTOR = 9
    ENUM = 10
    INTERFACE = 11
    FUNCTION = 12
    VARIABLE = 13
    CONSTANT = 14
    STRING = 15
    NUMBER = 16
    BOOLEAN = 17
    ARRAY = 18
    OBJECT = 19
    KEY = 20
    NULL = 21
    ENUM_MEMBER = 22
    STRUCT = 23
    EVENT = 24
    OPERATOR = 25
    TYPE_PARAMETER = 26


class CompletionItemKind(IntEnum):
    """Kind of a completion item."""
    TEXT = 1
    METHOD = 2
    FUNCTION = 3
    CONSTRUCTOR = 4
    FIELD = 5
    VARIABLE = 6
    CLASS = 7
    INTERFACE = 8
    MODULE = 9
    PROPERTY = 10
    UNIT = 11
    VALUE = 12
    ENUM = 13
    KEYWORD = 14
    SNIPPET = 15
    COLOR = 16
    FILE = 17
    REFERENCE = 18
    FOLDER = 19
    ENUM_MEMBER = 20
    CONSTANT = 21
    STRUCT = 22
    EVENT = 23
    OPERATOR = 24
    TYPE_PARAMETER = 25


@dataclass
class Position:
    """Position in a text document (0-indexed)."""
    line: int
    character: int
    
    def to_dict(self) -> dict[str, int]:
        return {"line": self.line, "character": self.character}
    
    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Position":
        return cls(line=data["line"], character=data["character"])


@dataclass
class Range:
    """A range in a text document."""
    start: Position
    end: Position
    
    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Range":
        return cls(
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
        )


@dataclass
class Location:
    """A location in a document."""
    uri: str
    range: Range
    
    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "range": self.range.to_dict()}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Location":
        return cls(uri=data["uri"], range=Range.from_dict(data["range"]))
    
    @property
    def file_path(self) -> str:
        """Get the file path from the URI."""
        if self.uri.startswith("file://"):
            return self.uri[7:]
        return self.uri


@dataclass
class Diagnostic:
    """A diagnostic (error, warning, etc.)."""
    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    code: str | int | None = None
    source: str | None = None
    related_information: list["DiagnosticRelatedInformation"] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "range": self.range.to_dict(),
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.code is not None:
            result["code"] = self.code
        if self.source is not None:
            result["source"] = self.source
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Diagnostic":
        related = []
        if "relatedInformation" in data:
            related = [
                DiagnosticRelatedInformation.from_dict(r)
                for r in data["relatedInformation"]
            ]
        return cls(
            range=Range.from_dict(data["range"]),
            message=data["message"],
            severity=DiagnosticSeverity(data.get("severity", 1)),
            code=data.get("code"),
            source=data.get("source"),
            related_information=related,
        )


@dataclass
class DiagnosticRelatedInformation:
    """Related information for a diagnostic."""
    location: Location
    message: str
    
    def to_dict(self) -> dict[str, Any]:
        return {"location": self.location.to_dict(), "message": self.message}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiagnosticRelatedInformation":
        return cls(
            location=Location.from_dict(data["location"]),
            message=data["message"],
        )


@dataclass
class TextEdit:
    """A text edit."""
    range: Range
    new_text: str
    
    def to_dict(self) -> dict[str, Any]:
        return {"range": self.range.to_dict(), "newText": self.new_text}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextEdit":
        return cls(
            range=Range.from_dict(data["range"]),
            new_text=data["newText"],
        )


@dataclass
class CompletionItem:
    """A completion item."""
    label: str
    kind: CompletionItemKind | None = None
    detail: str | None = None
    documentation: str | None = None
    deprecated: bool = False
    insert_text: str | None = None
    text_edit: TextEdit | None = None
    sort_text: str | None = None
    filter_text: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"label": self.label}
        if self.kind is not None:
            result["kind"] = self.kind.value
        if self.detail is not None:
            result["detail"] = self.detail
        if self.documentation is not None:
            result["documentation"] = self.documentation
        if self.deprecated:
            result["deprecated"] = True
        if self.insert_text is not None:
            result["insertText"] = self.insert_text
        if self.text_edit is not None:
            result["textEdit"] = self.text_edit.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompletionItem":
        text_edit = None
        if "textEdit" in data:
            text_edit = TextEdit.from_dict(data["textEdit"])
        return cls(
            label=data["label"],
            kind=CompletionItemKind(data["kind"]) if "kind" in data else None,
            detail=data.get("detail"),
            documentation=data.get("documentation") if isinstance(data.get("documentation"), str) else None,
            deprecated=data.get("deprecated", False),
            insert_text=data.get("insertText"),
            text_edit=text_edit,
            sort_text=data.get("sortText"),
            filter_text=data.get("filterText"),
        )


@dataclass
class Hover:
    """Hover information."""
    contents: str
    range: Range | None = None
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hover":
        contents = data.get("contents", "")
        if isinstance(contents, dict):
            contents = contents.get("value", str(contents))
        elif isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", str(item)))
            contents = "\n".join(parts)
        
        range_data = data.get("range")
        return cls(
            contents=str(contents),
            range=Range.from_dict(range_data) if range_data else None,
        )


@dataclass
class SymbolInformation:
    """Information about a symbol."""
    name: str
    kind: SymbolKind
    location: Location
    container_name: str | None = None
    deprecated: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "location": self.location.to_dict(),
        }
        if self.container_name is not None:
            result["containerName"] = self.container_name
        if self.deprecated:
            result["deprecated"] = True
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolInformation":
        return cls(
            name=data["name"],
            kind=SymbolKind(data["kind"]),
            location=Location.from_dict(data["location"]),
            container_name=data.get("containerName"),
            deprecated=data.get("deprecated", False),
        )


@dataclass
class DocumentSymbol:
    """A document symbol (hierarchical)."""
    name: str
    kind: SymbolKind
    range: Range
    selection_range: Range
    detail: str | None = None
    children: list["DocumentSymbol"] = field(default_factory=list)
    deprecated: bool = False
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentSymbol":
        children = []
        if "children" in data:
            children = [DocumentSymbol.from_dict(c) for c in data["children"]]
        return cls(
            name=data["name"],
            kind=SymbolKind(data["kind"]),
            range=Range.from_dict(data["range"]),
            selection_range=Range.from_dict(data["selectionRange"]),
            detail=data.get("detail"),
            children=children,
            deprecated=data.get("deprecated", False),
        )


@dataclass
class TextDocumentIdentifier:
    """Identifies a text document."""
    uri: str
    
    def to_dict(self) -> dict[str, str]:
        return {"uri": self.uri}


@dataclass
class TextDocumentItem:
    """An item to transfer a text document from client to server."""
    uri: str
    language_id: str
    version: int
    text: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "languageId": self.language_id,
            "version": self.version,
            "text": self.text,
        }


@dataclass
class VersionedTextDocumentIdentifier:
    """Identifies a versioned text document."""
    uri: str
    version: int
    
    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "version": self.version}


@dataclass
class TextDocumentContentChangeEvent:
    """A change to a text document."""
    text: str
    range: Range | None = None
    
    def to_dict(self) -> dict[str, Any]:
        if self.range is None:
            return {"text": self.text}
        return {"range": self.range.to_dict(), "text": self.text}


@dataclass
class WorkspaceFolder:
    """A workspace folder."""
    uri: str
    name: str
    
    def to_dict(self) -> dict[str, str]:
        return {"uri": self.uri, "name": self.name}


@dataclass
class ServerCapabilities:
    """Server capabilities."""
    text_document_sync: int | dict[str, Any] | None = None
    completion_provider: dict[str, Any] | None = None
    hover_provider: bool = False
    definition_provider: bool = False
    references_provider: bool = False
    document_symbol_provider: bool = False
    workspace_symbol_provider: bool = False
    code_action_provider: bool = False
    rename_provider: bool = False
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerCapabilities":
        return cls(
            text_document_sync=data.get("textDocumentSync"),
            completion_provider=data.get("completionProvider"),
            hover_provider=bool(data.get("hoverProvider")),
            definition_provider=bool(data.get("definitionProvider")),
            references_provider=bool(data.get("referencesProvider")),
            document_symbol_provider=bool(data.get("documentSymbolProvider")),
            workspace_symbol_provider=bool(data.get("workspaceSymbolProvider")),
            code_action_provider=bool(data.get("codeActionProvider")),
            rename_provider=bool(data.get("renameProvider")),
        )
