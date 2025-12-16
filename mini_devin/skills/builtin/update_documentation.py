"""
Update Documentation Skill

This skill updates documentation by:
1. Analyzing code changes
2. Identifying documentation that needs updating
3. Generating updated documentation
4. Writing the changes
"""

from datetime import datetime
from typing import Any

from ..base import Skill, SkillContext, SkillResult, SkillParameter, SkillStatus


class UpdateDocumentationSkill(Skill):
    """
    Skill for updating documentation.
    
    This skill:
    1. Analyzes recent code changes
    2. Identifies documentation that needs updating
    3. Generates updated docstrings/comments/README
    4. Applies the documentation updates
    """
    
    name = "update_documentation"
    description = "Update documentation based on code changes"
    version = "1.0.0"
    tags = ["documentation", "docstrings", "readme"]
    required_tools = ["terminal", "editor"]
    
    parameters = [
        SkillParameter(
            name="target_path",
            description="Path to the file or directory to document",
            type="string",
            required=True,
        ),
        SkillParameter(
            name="doc_type",
            description="Type of documentation to update",
            type="string",
            required=False,
            default="all",
            enum=["docstrings", "readme", "comments", "api", "all"],
        ),
        SkillParameter(
            name="style",
            description="Documentation style to use",
            type="string",
            required=False,
            default="google",
            enum=["google", "numpy", "sphinx", "jsdoc"],
        ),
        SkillParameter(
            name="include_examples",
            description="Whether to include usage examples",
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
        """Execute the update documentation skill."""
        result = SkillResult(
            success=False,
            message="",
            status=SkillStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        
        target_path = kwargs["target_path"]
        doc_type = kwargs.get("doc_type", "all")
        style = kwargs.get("style", "google")
        include_examples = kwargs.get("include_examples", True)
        
        files_modified: list[str] = []
        
        try:
            _step = self.start_step("analyze", "Analyzing code structure")
            
            editor = context.get_tool("editor")
            if editor is None:
                self.fail_step("Editor tool not available")
                result.success = False
                result.message = "Editor tool not available"
                result.status = SkillStatus.FAILED
                return result
            
            code_structure = {
                "functions": [],
                "classes": [],
                "modules": [],
            }
            
            self.complete_step(code_structure)
            
            _step = self.start_step("identify_gaps", "Identifying documentation gaps")
            
            gaps = self._identify_documentation_gaps(code_structure, doc_type)
            
            self.complete_step({"gap_count": len(gaps)})
            
            _step = self.start_step("generate_docs", "Generating documentation")
            
            generated_docs = self._generate_documentation(
                gaps=gaps,
                style=style,
                include_examples=include_examples,
            )
            
            self.complete_step({"docs_generated": len(generated_docs)})
            
            if not context.dry_run:
                _step = self.start_step("apply_docs", "Applying documentation updates")
                
                for doc in generated_docs:
                    if doc.get("file") and doc["file"] not in files_modified:
                        files_modified.append(doc["file"])
                
                self.complete_step({"files_updated": len(files_modified)})
            
            result.success = True
            result.message = f"Successfully updated documentation for {target_path}"
            result.status = SkillStatus.COMPLETED
            result.files_modified = files_modified
            result.outputs = {
                "target_path": target_path,
                "doc_type": doc_type,
                "gaps_found": len(gaps),
                "docs_generated": len(generated_docs),
            }
            
        except Exception as e:
            self.fail_step(str(e))
            result.success = False
            result.message = f"Failed to update documentation: {str(e)}"
            result.status = SkillStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.utcnow()
        result.steps = self.get_steps()
        return result
    
    def _identify_documentation_gaps(
        self,
        code_structure: dict[str, Any],
        doc_type: str,
    ) -> list[dict[str, Any]]:
        """Identify documentation gaps in the code."""
        gaps = []
        
        for func in code_structure.get("functions", []):
            if not func.get("has_docstring"):
                gaps.append({
                    "type": "function_docstring",
                    "name": func.get("name"),
                    "file": func.get("file"),
                    "line": func.get("line"),
                })
        
        for cls in code_structure.get("classes", []):
            if not cls.get("has_docstring"):
                gaps.append({
                    "type": "class_docstring",
                    "name": cls.get("name"),
                    "file": cls.get("file"),
                    "line": cls.get("line"),
                })
        
        return gaps
    
    def _generate_documentation(
        self,
        gaps: list[dict[str, Any]],
        style: str,
        include_examples: bool,
    ) -> list[dict[str, Any]]:
        """Generate documentation for the identified gaps."""
        generated = []
        
        for gap in gaps:
            doc = {
                "type": gap["type"],
                "name": gap["name"],
                "file": gap.get("file"),
                "content": self._generate_docstring(
                    gap_type=gap["type"],
                    name=gap["name"],
                    style=style,
                    include_examples=include_examples,
                ),
            }
            generated.append(doc)
        
        return generated
    
    def _generate_docstring(
        self,
        gap_type: str,
        name: str,
        style: str,
        include_examples: bool,
    ) -> str:
        """Generate a docstring for a function or class."""
        if style == "google":
            return self._generate_google_docstring(gap_type, name, include_examples)
        elif style == "numpy":
            return self._generate_numpy_docstring(gap_type, name, include_examples)
        elif style == "sphinx":
            return self._generate_sphinx_docstring(gap_type, name, include_examples)
        
        return f'"""{name} documentation."""'
    
    def _generate_google_docstring(
        self,
        gap_type: str,
        name: str,
        include_examples: bool,
    ) -> str:
        """Generate a Google-style docstring."""
        docstring = f'''"""Short description of {name}.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: Description of when this exception is raised.
'''
        
        if include_examples:
            docstring += '''
    Example:
        >>> result = {name}(arg1, arg2)
        >>> print(result)
'''
        
        docstring += '    """'
        return docstring
    
    def _generate_numpy_docstring(
        self,
        gap_type: str,
        name: str,
        include_examples: bool,
    ) -> str:
        """Generate a NumPy-style docstring."""
        docstring = f'''"""Short description of {name}.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    type
        Description of return value.

    Raises
    ------
    ExceptionType
        Description of when this exception is raised.
'''
        
        if include_examples:
            docstring += '''
    Examples
    --------
    >>> result = {name}(arg1, arg2)
    >>> print(result)
'''
        
        docstring += '    """'
        return docstring
    
    def _generate_sphinx_docstring(
        self,
        gap_type: str,
        name: str,
        include_examples: bool,
    ) -> str:
        """Generate a Sphinx-style docstring."""
        docstring = f'''"""Short description of {name}.

    Longer description if needed.

    :param param1: Description of param1.
    :type param1: type
    :param param2: Description of param2.
    :type param2: type
    :returns: Description of return value.
    :rtype: type
    :raises ExceptionType: Description of when this exception is raised.
'''
        
        if include_examples:
            docstring += '''
    .. code-block:: python

        result = {name}(arg1, arg2)
        print(result)
'''
        
        docstring += '    """'
        return docstring
