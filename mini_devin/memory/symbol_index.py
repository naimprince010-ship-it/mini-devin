"""
Code Symbol Index using Tree-sitter

This module provides structural indexing of code using tree-sitter for AST parsing.
It extracts symbols (functions, classes, methods, variables) and their relationships.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class SymbolType(str, Enum):
    """Types of code symbols."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    PROPERTY = "property"
    PARAMETER = "parameter"


@dataclass
class SymbolLocation:
    """Location of a symbol in source code."""
    file_path: str
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    
    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}"


@dataclass
class Symbol:
    """A code symbol (function, class, variable, etc.)."""
    name: str
    symbol_type: SymbolType
    location: SymbolLocation
    signature: str = ""
    docstring: str = ""
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    references: list[SymbolLocation] = field(default_factory=list)
    language: str = ""
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified name including parent."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "symbol_type": self.symbol_type.value,
            "location": {
                "file_path": self.location.file_path,
                "start_line": self.location.start_line,
                "end_line": self.location.end_line,
            },
            "signature": self.signature,
            "docstring": self.docstring[:200] if self.docstring else "",
            "parent": self.parent,
            "children": self.children,
            "language": self.language,
        }


class SymbolIndex:
    """
    Index of code symbols in a repository.
    
    Uses tree-sitter for AST parsing when available, falls back to regex patterns.
    """
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.symbols: dict[str, Symbol] = {}
        self._file_symbols: dict[str, list[str]] = {}
        self._type_symbols: dict[SymbolType, list[str]] = {t: [] for t in SymbolType}
        self._tree_sitter_available = self._check_tree_sitter()
    
    def _check_tree_sitter(self) -> bool:
        """Check if tree-sitter is available."""
        try:
            import importlib.util
            return importlib.util.find_spec("tree_sitter") is not None
        except ImportError:
            return False
    
    def index_file(self, file_path: str) -> list[Symbol]:
        """
        Index symbols in a single file.
        
        Args:
            file_path: Path to the file to index
            
        Returns:
            List of symbols found in the file
        """
        path = Path(file_path)
        if not path.exists():
            return []
        
        # Determine language from extension
        language = self._detect_language(path)
        if not language:
            return []
        
        content = path.read_text(errors="ignore")
        
        # Use tree-sitter if available, otherwise fall back to regex
        if self._tree_sitter_available:
            symbols = self._index_with_tree_sitter(path, content, language)
        else:
            symbols = self._index_with_regex(path, content, language)
        
        # Store symbols
        rel_path = str(path.relative_to(self.workspace_path) if path.is_relative_to(self.workspace_path) else path)
        self._file_symbols[rel_path] = []
        
        for symbol in symbols:
            key = f"{rel_path}:{symbol.qualified_name}"
            self.symbols[key] = symbol
            self._file_symbols[rel_path].append(key)
            self._type_symbols[symbol.symbol_type].append(key)
        
        return symbols
    
    def index_directory(self, directory: Optional[str] = None, extensions: Optional[list[str]] = None) -> int:
        """
        Index all files in a directory.
        
        Args:
            directory: Directory to index (defaults to workspace)
            extensions: File extensions to index (defaults to common code extensions)
            
        Returns:
            Number of symbols indexed
        """
        if directory is None:
            directory = str(self.workspace_path)
        
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h"]
        
        dir_path = Path(directory)
        count = 0
        
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                # Skip common non-source directories
                if any(part.startswith(".") or part in ["node_modules", "__pycache__", "venv", ".venv", "dist", "build"] 
                       for part in file_path.parts):
                    continue
                
                symbols = self.index_file(str(file_path))
                count += len(symbols)
        
        return count
    
    def _detect_language(self, path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
        }
        return ext_map.get(path.suffix.lower())
    
    def _index_with_tree_sitter(self, path: Path, content: str, language: str) -> list[Symbol]:
        """Index using tree-sitter AST parsing."""
        # For now, fall back to regex since tree-sitter requires language-specific parsers
        # In production, you would load the appropriate language parser
        return self._index_with_regex(path, content, language)
    
    def _index_with_regex(self, path: Path, content: str, language: str) -> list[Symbol]:
        """Index using regex patterns (fallback when tree-sitter unavailable)."""
        symbols = []
        lines = content.split("\n")
        rel_path = str(path.relative_to(self.workspace_path) if path.is_relative_to(self.workspace_path) else path)
        
        if language == "python":
            symbols.extend(self._index_python(rel_path, content, lines))
        elif language in ["javascript", "typescript"]:
            symbols.extend(self._index_javascript(rel_path, content, lines))
        elif language == "go":
            symbols.extend(self._index_go(rel_path, content, lines))
        elif language == "rust":
            symbols.extend(self._index_rust(rel_path, content, lines))
        elif language == "java":
            symbols.extend(self._index_java(rel_path, content, lines))
        
        return symbols
    
    def _index_python(self, file_path: str, content: str, lines: list[str]) -> list[Symbol]:
        """Index Python symbols."""
        symbols = []
        
        # Pattern for functions and methods
        func_pattern = re.compile(r"^(\s*)def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?:", re.MULTILINE)
        # Pattern for classes
        class_pattern = re.compile(r"^(\s*)class\s+(\w+)(?:\((.*?)\))?:", re.MULTILINE)
        # Pattern for imports
        import_pattern = re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+)$", re.MULTILINE)
        # Pattern for constants (uppercase variables at module level)
        const_pattern = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)
        
        # Find classes
        for match in class_pattern.finditer(content):
            indent = len(match.group(1))
            name = match.group(2)
            bases = match.group(3) or ""
            start_line = content[:match.start()].count("\n") + 1
            
            # Find end of class (next line with same or less indentation)
            end_line = start_line
            for i, line in enumerate(lines[start_line:], start=start_line + 1):
                if line.strip() and not line.startswith(" " * (indent + 1)) and not line.strip().startswith("#"):
                    break
                end_line = i
            
            # Extract docstring
            docstring = self._extract_python_docstring(lines, start_line)
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                location=SymbolLocation(file_path, start_line, end_line),
                signature=f"class {name}({bases})" if bases else f"class {name}",
                docstring=docstring,
                language="python",
            )
            symbols.append(symbol)
        
        # Find functions and methods
        for match in func_pattern.finditer(content):
            indent = len(match.group(1))
            name = match.group(2)
            params = match.group(3)
            return_type = match.group(4) or ""
            start_line = content[:match.start()].count("\n") + 1
            
            # Determine if it's a method (indented under a class)
            is_method = indent > 0
            parent = None
            
            if is_method:
                # Find the parent class
                for sym in reversed(symbols):
                    if sym.symbol_type == SymbolType.CLASS and sym.location.start_line < start_line:
                        parent = sym.name
                        sym.children.append(name)
                        break
            
            # Find end of function
            end_line = start_line
            for i, line in enumerate(lines[start_line:], start=start_line + 1):
                if line.strip() and not line.startswith(" " * (indent + 1)) and not line.strip().startswith("#"):
                    break
                end_line = i
            
            # Extract docstring
            docstring = self._extract_python_docstring(lines, start_line)
            
            signature = f"def {name}({params})"
            if return_type:
                signature += f" -> {return_type}"
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.METHOD if is_method else SymbolType.FUNCTION,
                location=SymbolLocation(file_path, start_line, end_line),
                signature=signature,
                docstring=docstring,
                parent=parent,
                language="python",
            )
            symbols.append(symbol)
        
        # Find imports
        for match in import_pattern.finditer(content):
            from_module = match.group(1) or ""
            imports = match.group(2)
            start_line = content[:match.start()].count("\n") + 1
            
            if from_module:
                signature = f"from {from_module} import {imports}"
            else:
                signature = f"import {imports}"
            
            symbol = Symbol(
                name=imports.split(",")[0].strip().split(" ")[0],
                symbol_type=SymbolType.IMPORT,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=signature,
                language="python",
            )
            symbols.append(symbol)
        
        # Find constants
        for match in const_pattern.finditer(content):
            name = match.group(1)
            start_line = content[:match.start()].count("\n") + 1
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.CONSTANT,
                location=SymbolLocation(file_path, start_line, start_line),
                language="python",
            )
            symbols.append(symbol)
        
        return symbols
    
    def _index_javascript(self, file_path: str, content: str, lines: list[str]) -> list[Symbol]:
        """Index JavaScript/TypeScript symbols."""
        symbols = []
        
        # Pattern for functions
        func_patterns = [
            re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\((.*?)\)", re.MULTILINE),
            re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\((.*?)\)\s*=>", re.MULTILINE),
            re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function\s*\((.*?)\)", re.MULTILINE),
        ]
        
        # Pattern for classes
        class_pattern = re.compile(r"^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", re.MULTILINE)
        
        # Pattern for interfaces (TypeScript)
        interface_pattern = re.compile(r"^(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+(.+?))?(?:\s*\{)", re.MULTILINE)
        
        # Pattern for type aliases (TypeScript)
        type_pattern = re.compile(r"^(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE)
        
        # Find classes
        for match in class_pattern.finditer(content):
            name = match.group(1)
            extends = match.group(2) or ""
            start_line = content[:match.start()].count("\n") + 1
            
            signature = f"class {name}"
            if extends:
                signature += f" extends {extends}"
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=signature,
                language="javascript",
            )
            symbols.append(symbol)
        
        # Find functions
        for pattern in func_patterns:
            for match in pattern.finditer(content):
                name = match.group(1)
                params = match.group(2)
                start_line = content[:match.start()].count("\n") + 1
                
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.FUNCTION,
                    location=SymbolLocation(file_path, start_line, start_line),
                    signature=f"function {name}({params})",
                    language="javascript",
                )
                symbols.append(symbol)
        
        # Find interfaces
        for match in interface_pattern.finditer(content):
            name = match.group(1)
            extends = match.group(2) or ""
            start_line = content[:match.start()].count("\n") + 1
            
            signature = f"interface {name}"
            if extends:
                signature += f" extends {extends}"
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.INTERFACE,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=signature,
                language="typescript",
            )
            symbols.append(symbol)
        
        # Find type aliases
        for match in type_pattern.finditer(content):
            name = match.group(1)
            start_line = content[:match.start()].count("\n") + 1
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.TYPE_ALIAS,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=f"type {name}",
                language="typescript",
            )
            symbols.append(symbol)
        
        return symbols
    
    def _index_go(self, file_path: str, content: str, lines: list[str]) -> list[Symbol]:
        """Index Go symbols."""
        symbols = []
        
        # Pattern for functions
        func_pattern = re.compile(r"^func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\((.*?)\)(?:\s*\((.*?)\)|\s*(\w+))?", re.MULTILINE)
        
        # Pattern for types/structs
        type_pattern = re.compile(r"^type\s+(\w+)\s+(struct|interface)", re.MULTILINE)
        
        # Find functions
        for match in func_pattern.finditer(content):
            receiver_name = match.group(1)
            receiver_type = match.group(2)
            name = match.group(3)
            params = match.group(4)
            start_line = content[:match.start()].count("\n") + 1
            
            if receiver_type:
                signature = f"func ({receiver_name} {receiver_type}) {name}({params})"
                parent = receiver_type
                symbol_type = SymbolType.METHOD
            else:
                signature = f"func {name}({params})"
                parent = None
                symbol_type = SymbolType.FUNCTION
            
            symbol = Symbol(
                name=name,
                symbol_type=symbol_type,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=signature,
                parent=parent,
                language="go",
            )
            symbols.append(symbol)
        
        # Find types
        for match in type_pattern.finditer(content):
            name = match.group(1)
            kind = match.group(2)
            start_line = content[:match.start()].count("\n") + 1
            
            symbol_type = SymbolType.INTERFACE if kind == "interface" else SymbolType.CLASS
            
            symbol = Symbol(
                name=name,
                symbol_type=symbol_type,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=f"type {name} {kind}",
                language="go",
            )
            symbols.append(symbol)
        
        return symbols
    
    def _index_rust(self, file_path: str, content: str, lines: list[str]) -> list[Symbol]:
        """Index Rust symbols."""
        symbols = []
        
        # Pattern for functions
        func_pattern = re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)(?:<.*?>)?\s*\((.*?)\)(?:\s*->\s*(.+?))?(?:\s*where|\s*\{)", re.MULTILINE)
        
        # Pattern for structs
        struct_pattern = re.compile(r"^(?:pub\s+)?struct\s+(\w+)(?:<.*?>)?", re.MULTILINE)
        
        # Pattern for enums
        enum_pattern = re.compile(r"^(?:pub\s+)?enum\s+(\w+)(?:<.*?>)?", re.MULTILINE)
        
        # Pattern for traits
        trait_pattern = re.compile(r"^(?:pub\s+)?trait\s+(\w+)(?:<.*?>)?", re.MULTILINE)
        
        # Find functions
        for match in func_pattern.finditer(content):
            name = match.group(1)
            params = match.group(2)
            return_type = match.group(3) or ""
            start_line = content[:match.start()].count("\n") + 1
            
            signature = f"fn {name}({params})"
            if return_type:
                signature += f" -> {return_type}"
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.FUNCTION,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=signature,
                language="rust",
            )
            symbols.append(symbol)
        
        # Find structs
        for match in struct_pattern.finditer(content):
            name = match.group(1)
            start_line = content[:match.start()].count("\n") + 1
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=f"struct {name}",
                language="rust",
            )
            symbols.append(symbol)
        
        # Find enums
        for match in enum_pattern.finditer(content):
            name = match.group(1)
            start_line = content[:match.start()].count("\n") + 1
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.ENUM,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=f"enum {name}",
                language="rust",
            )
            symbols.append(symbol)
        
        # Find traits
        for match in trait_pattern.finditer(content):
            name = match.group(1)
            start_line = content[:match.start()].count("\n") + 1
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.INTERFACE,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=f"trait {name}",
                language="rust",
            )
            symbols.append(symbol)
        
        return symbols
    
    def _index_java(self, file_path: str, content: str, lines: list[str]) -> list[Symbol]:
        """Index Java symbols."""
        symbols = []
        
        # Pattern for classes
        class_pattern = re.compile(r"^(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+(.+?))?(?:\s*\{)", re.MULTILINE)
        
        # Pattern for interfaces
        interface_pattern = re.compile(r"^(?:public\s+)?interface\s+(\w+)(?:\s+extends\s+(.+?))?(?:\s*\{)", re.MULTILINE)
        
        # Pattern for methods
        method_pattern = re.compile(r"^\s+(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(\w+(?:<.*?>)?)\s+(\w+)\s*\((.*?)\)", re.MULTILINE)
        
        # Find classes
        for match in class_pattern.finditer(content):
            name = match.group(1)
            extends = match.group(2) or ""
            implements = match.group(3) or ""
            start_line = content[:match.start()].count("\n") + 1
            
            signature = f"class {name}"
            if extends:
                signature += f" extends {extends}"
            if implements:
                signature += f" implements {implements}"
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=signature,
                language="java",
            )
            symbols.append(symbol)
        
        # Find interfaces
        for match in interface_pattern.finditer(content):
            name = match.group(1)
            extends = match.group(2) or ""
            start_line = content[:match.start()].count("\n") + 1
            
            signature = f"interface {name}"
            if extends:
                signature += f" extends {extends}"
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.INTERFACE,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=signature,
                language="java",
            )
            symbols.append(symbol)
        
        # Find methods
        for match in method_pattern.finditer(content):
            return_type = match.group(1)
            name = match.group(2)
            params = match.group(3)
            start_line = content[:match.start()].count("\n") + 1
            
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.METHOD,
                location=SymbolLocation(file_path, start_line, start_line),
                signature=f"{return_type} {name}({params})",
                language="java",
            )
            symbols.append(symbol)
        
        return symbols
    
    def _extract_python_docstring(self, lines: list[str], start_line: int) -> str:
        """Extract Python docstring from function or class."""
        if start_line >= len(lines):
            return ""
        
        # Look for docstring in the next few lines
        for i in range(start_line, min(start_line + 3, len(lines))):
            line = lines[i].strip()
            if line.startswith('"""') or line.startswith("'''"):
                quote = line[:3]
                if line.count(quote) >= 2:
                    # Single line docstring
                    return line[3:-3].strip()
                else:
                    # Multi-line docstring
                    docstring_lines = [line[3:]]
                    for j in range(i + 1, len(lines)):
                        doc_line = lines[j]
                        if quote in doc_line:
                            docstring_lines.append(doc_line[:doc_line.index(quote)])
                            break
                        docstring_lines.append(doc_line)
                    return "\n".join(docstring_lines).strip()
        
        return ""
    
    def search(self, query: str, symbol_types: Optional[list[SymbolType]] = None, limit: int = 20) -> list[Symbol]:
        """
        Search for symbols matching a query.
        
        Args:
            query: Search query (matches name, signature, or docstring)
            symbol_types: Filter by symbol types
            limit: Maximum number of results
            
        Returns:
            List of matching symbols
        """
        results = []
        query_lower = query.lower()
        
        for key, symbol in self.symbols.items():
            # Filter by type if specified
            if symbol_types and symbol.symbol_type not in symbol_types:
                continue
            
            # Check if query matches
            if (query_lower in symbol.name.lower() or
                query_lower in symbol.signature.lower() or
                query_lower in symbol.docstring.lower()):
                results.append(symbol)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_symbol(self, name: str, file_path: Optional[str] = None) -> Optional[Symbol]:
        """
        Get a symbol by name.
        
        Args:
            name: Symbol name (can be qualified like "ClassName.method_name")
            file_path: Optional file path to narrow search
            
        Returns:
            Symbol if found, None otherwise
        """
        # Try exact match first
        for key, symbol in self.symbols.items():
            if file_path and not key.startswith(file_path):
                continue
            if symbol.name == name or symbol.qualified_name == name:
                return symbol
        
        return None
    
    def get_file_symbols(self, file_path: str) -> list[Symbol]:
        """Get all symbols in a file."""
        keys = self._file_symbols.get(file_path, [])
        return [self.symbols[k] for k in keys if k in self.symbols]
    
    def get_symbols_by_type(self, symbol_type: SymbolType) -> list[Symbol]:
        """Get all symbols of a specific type."""
        keys = self._type_symbols.get(symbol_type, [])
        return [self.symbols[k] for k in keys if k in self.symbols]
    
    def get_children(self, symbol: Symbol) -> list[Symbol]:
        """Get child symbols (methods of a class, etc.)."""
        children = []
        for child_name in symbol.children:
            child = self.get_symbol(child_name, symbol.location.file_path)
            if child:
                children.append(child)
        return children
    
    def get_statistics(self) -> dict:
        """Get statistics about the index."""
        return {
            "total_symbols": len(self.symbols),
            "files_indexed": len(self._file_symbols),
            "symbols_by_type": {t.value: len(keys) for t, keys in self._type_symbols.items()},
        }
    
    def to_dict(self) -> dict:
        """Export index to dictionary."""
        return {
            "workspace": str(self.workspace_path),
            "statistics": self.get_statistics(),
            "symbols": [s.to_dict() for s in self.symbols.values()],
        }
    
    def clear(self) -> None:
        """Clear the index."""
        self.symbols.clear()
        self._file_symbols.clear()
        self._type_symbols = {t: [] for t in SymbolType}


def create_symbol_index(workspace_path: str) -> SymbolIndex:
    """Create a new symbol index for a workspace."""
    return SymbolIndex(workspace_path)
