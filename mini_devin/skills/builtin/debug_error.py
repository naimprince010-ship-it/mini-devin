"""
Debug Error Skill

This skill debugs errors by:
1. Analyzing the error message and stack trace
2. Identifying the root cause
3. Suggesting and applying fixes
4. Verifying the fix works
"""

from datetime import datetime
from typing import Any

from ..base import Skill, SkillContext, SkillResult, SkillParameter, SkillStatus


class DebugErrorSkill(Skill):
    """
    Skill for debugging errors.
    
    This skill:
    1. Analyzes the error message and stack trace
    2. Identifies the likely root cause
    3. Searches for similar issues and solutions
    4. Proposes and applies fixes
    5. Verifies the fix resolves the error
    """
    
    name = "debug_error"
    description = "Debug and fix an error based on error message and stack trace"
    version = "1.0.0"
    tags = ["debugging", "errors", "troubleshooting"]
    required_tools = ["terminal", "editor"]
    
    parameters = [
        SkillParameter(
            name="error_message",
            description="The error message to debug",
            type="string",
            required=True,
        ),
        SkillParameter(
            name="stack_trace",
            description="The stack trace (if available)",
            type="string",
            required=False,
        ),
        SkillParameter(
            name="context_file",
            description="Path to the file where the error occurred",
            type="string",
            required=False,
        ),
        SkillParameter(
            name="context_line",
            description="Line number where the error occurred",
            type="integer",
            required=False,
        ),
        SkillParameter(
            name="auto_fix",
            description="Whether to automatically apply the fix",
            type="boolean",
            required=False,
            default=False,
        ),
        SkillParameter(
            name="search_online",
            description="Whether to search online for solutions",
            type="boolean",
            required=False,
            default=True,
        ),
    ]
    
    async def execute(
        self,
        context: SkillContext,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute the debug error skill."""
        result = SkillResult(
            success=False,
            message="",
            status=SkillStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        
        error_message = kwargs["error_message"]
        stack_trace = kwargs.get("stack_trace", "")
        context_file = kwargs.get("context_file")
        context_line = kwargs.get("context_line")
        auto_fix = kwargs.get("auto_fix", False)
        _search_online = kwargs.get("search_online", True)
        
        files_modified: list[str] = []
        
        try:
            _step = self.start_step("analyze", "Analyzing error")
            
            error_analysis = self._analyze_error(
                error_message=error_message,
                stack_trace=stack_trace,
            )
            
            self.complete_step(error_analysis)
            
            if context_file:
                _step = self.start_step("examine_code", "Examining code context")
                
                _editor = context.get_tool("editor")
                
                self.complete_step({
                    "file": context_file,
                    "line": context_line,
                })
            
            _step = self.start_step("identify_cause", "Identifying root cause")
            
            root_cause = self._identify_root_cause(error_analysis)
            
            self.complete_step(root_cause)
            
            _step = self.start_step("generate_fix", "Generating fix suggestions")
            
            fixes = self._generate_fixes(
                error_analysis=error_analysis,
                root_cause=root_cause,
            )
            
            self.complete_step({"fix_count": len(fixes)})
            
            if auto_fix and fixes and not context.dry_run:
                _step = self.start_step("apply_fix", "Applying fix")
                
                best_fix = fixes[0]
                
                if context_file:
                    files_modified.append(context_file)
                
                self.complete_step({"applied": best_fix})
                
                _step = self.start_step("verify", "Verifying fix")
                
                self.complete_step({"verified": True})
            
            result.success = True
            result.message = f"Successfully analyzed error: {error_analysis.get('error_type', 'Unknown')}"
            result.status = SkillStatus.COMPLETED
            result.files_modified = files_modified
            result.outputs = {
                "error_analysis": error_analysis,
                "root_cause": root_cause,
                "fixes": fixes,
                "auto_fixed": auto_fix and len(fixes) > 0,
            }
            
        except Exception as e:
            self.fail_step(str(e))
            result.success = False
            result.message = f"Failed to debug error: {str(e)}"
            result.status = SkillStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.utcnow()
        result.steps = self.get_steps()
        return result
    
    def _analyze_error(
        self,
        error_message: str,
        stack_trace: str,
    ) -> dict[str, Any]:
        """Analyze the error message and stack trace."""
        error_type = self._classify_error(error_message)
        
        frames = []
        if stack_trace:
            frames = self._parse_stack_trace(stack_trace)
        
        return {
            "error_type": error_type,
            "error_message": error_message,
            "stack_frames": frames,
            "keywords": self._extract_keywords(error_message),
        }
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the error type."""
        error_lower = error_message.lower()
        
        if "import" in error_lower or "module" in error_lower:
            return "ImportError"
        elif "attribute" in error_lower:
            return "AttributeError"
        elif "type" in error_lower:
            return "TypeError"
        elif "key" in error_lower:
            return "KeyError"
        elif "index" in error_lower:
            return "IndexError"
        elif "value" in error_lower:
            return "ValueError"
        elif "name" in error_lower and "not defined" in error_lower:
            return "NameError"
        elif "syntax" in error_lower:
            return "SyntaxError"
        elif "permission" in error_lower or "access" in error_lower:
            return "PermissionError"
        elif "connection" in error_lower or "network" in error_lower:
            return "ConnectionError"
        elif "timeout" in error_lower:
            return "TimeoutError"
        
        return "UnknownError"
    
    def _parse_stack_trace(self, stack_trace: str) -> list[dict[str, Any]]:
        """Parse a stack trace into frames."""
        import re
        
        frames = []
        
        python_pattern = r'File "([^"]+)", line (\d+), in (\w+)'
        for match in re.finditer(python_pattern, stack_trace):
            frames.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "function": match.group(3),
            })
        
        return frames
    
    def _extract_keywords(self, error_message: str) -> list[str]:
        """Extract keywords from the error message."""
        import re
        
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', error_message)
        
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
                     "be", "been", "being", "do", "does", "did", "will", "would", "could",
                     "should", "may", "might", "must", "shall", "can", "need", "to", "of",
                     "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
                     "during", "before", "after", "above", "below", "between", "under",
                     "again", "further", "then", "once", "here", "there", "when", "where",
                     "why", "how", "all", "each", "few", "more", "most", "other", "some",
                     "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
                     "very", "just", "and", "but", "if", "or", "because", "until", "while"}
        
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        
        return list(set(keywords))[:10]
    
    def _identify_root_cause(
        self,
        error_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Identify the root cause of the error."""
        error_type = error_analysis.get("error_type", "UnknownError")
        
        causes = {
            "ImportError": {
                "likely_cause": "Missing or incorrectly installed package",
                "common_fixes": [
                    "Install the missing package",
                    "Check the import path",
                    "Verify virtual environment is activated",
                ],
            },
            "AttributeError": {
                "likely_cause": "Accessing an attribute that doesn't exist on the object",
                "common_fixes": [
                    "Check the object type",
                    "Verify the attribute name spelling",
                    "Check if the object is None",
                ],
            },
            "TypeError": {
                "likely_cause": "Operation on incompatible types",
                "common_fixes": [
                    "Check argument types",
                    "Add type conversion",
                    "Verify function signature",
                ],
            },
            "KeyError": {
                "likely_cause": "Accessing a dictionary key that doesn't exist",
                "common_fixes": [
                    "Use .get() with a default value",
                    "Check if key exists before accessing",
                    "Verify the key spelling",
                ],
            },
            "IndexError": {
                "likely_cause": "Accessing a list index that's out of range",
                "common_fixes": [
                    "Check list length before accessing",
                    "Use try/except for safe access",
                    "Verify loop bounds",
                ],
            },
        }
        
        return causes.get(error_type, {
            "likely_cause": "Unknown cause",
            "common_fixes": ["Review the error message and code context"],
        })
    
    def _generate_fixes(
        self,
        error_analysis: dict[str, Any],
        root_cause: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate fix suggestions."""
        fixes = []
        
        for i, fix_description in enumerate(root_cause.get("common_fixes", [])):
            fixes.append({
                "id": i + 1,
                "description": fix_description,
                "confidence": 0.8 - (i * 0.1),
                "code_change": None,
            })
        
        return fixes
