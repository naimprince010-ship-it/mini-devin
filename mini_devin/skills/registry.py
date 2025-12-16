"""
Skill Registry

This module provides a registry for managing and discovering skills.
Skills can be registered, discovered, and invoked through the registry.
"""

from typing import Any, Type
import importlib
from pathlib import Path

from .base import Skill, SkillContext, SkillResult


class SkillRegistry:
    """
    Registry for managing skills.
    
    Provides:
    - Skill registration and discovery
    - Skill lookup by name or tags
    - Skill invocation with validation
    """
    
    _instance: "SkillRegistry | None" = None
    
    def __init__(self):
        self._skills: dict[str, Type[Skill]] = {}
        self._instances: dict[str, Skill] = {}
        self._tags_index: dict[str, set[str]] = {}
    
    @classmethod
    def get_instance(cls) -> "SkillRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, skill_class: Type[Skill]) -> None:
        """
        Register a skill class.
        
        Args:
            skill_class: The skill class to register
        """
        name = skill_class.name
        self._skills[name] = skill_class
        
        for tag in skill_class.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(name)
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a skill by name.
        
        Args:
            name: Name of the skill to unregister
            
        Returns:
            True if the skill was unregistered, False if not found
        """
        if name not in self._skills:
            return False
        
        skill_class = self._skills[name]
        del self._skills[name]
        
        if name in self._instances:
            del self._instances[name]
        
        for tag in skill_class.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(name)
        
        return True
    
    def get(self, name: str) -> Skill | None:
        """
        Get a skill instance by name.
        
        Args:
            name: Name of the skill
            
        Returns:
            Skill instance or None if not found
        """
        if name not in self._skills:
            return None
        
        if name not in self._instances:
            self._instances[name] = self._skills[name]()
        
        return self._instances[name]
    
    def get_class(self, name: str) -> Type[Skill] | None:
        """
        Get a skill class by name.
        
        Args:
            name: Name of the skill
            
        Returns:
            Skill class or None if not found
        """
        return self._skills.get(name)
    
    def list_skills(self) -> list[str]:
        """Get a list of all registered skill names."""
        return list(self._skills.keys())
    
    def list_by_tag(self, tag: str) -> list[str]:
        """
        Get skills by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of skill names with the given tag
        """
        return list(self._tags_index.get(tag, set()))
    
    def list_tags(self) -> list[str]:
        """Get a list of all tags."""
        return list(self._tags_index.keys())
    
    def search(self, query: str) -> list[str]:
        """
        Search for skills by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching skill names
        """
        query_lower = query.lower()
        results = []
        
        for name, skill_class in self._skills.items():
            if query_lower in name.lower():
                results.append(name)
            elif query_lower in skill_class.description.lower():
                results.append(name)
        
        return results
    
    async def execute(
        self,
        name: str,
        context: SkillContext,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Execute a skill by name.
        
        Args:
            name: Name of the skill to execute
            context: Execution context
            **kwargs: Skill parameters
            
        Returns:
            SkillResult from the skill execution
        """
        skill = self.get(name)
        if skill is None:
            return SkillResult.failure_result(
                message=f"Skill not found: {name}",
                error="SKILL_NOT_FOUND",
            )
        
        valid, error = skill.validate_parameters(**kwargs)
        if not valid:
            return SkillResult.failure_result(
                message=f"Invalid parameters: {error}",
                error="INVALID_PARAMETERS",
            )
        
        valid, error = skill.validate_context(context)
        if not valid:
            return SkillResult.failure_result(
                message=f"Invalid context: {error}",
                error="INVALID_CONTEXT",
            )
        
        skill.reset()
        
        try:
            return await skill.execute(context, **kwargs)
        except Exception as e:
            return SkillResult.failure_result(
                message=f"Skill execution failed: {str(e)}",
                error="EXECUTION_ERROR",
                error_details={"exception": str(e), "type": type(e).__name__},
            )
    
    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all registered skills."""
        schemas = []
        for name in self._skills:
            skill = self.get(name)
            if skill:
                schemas.append(skill.to_schema())
        return schemas
    
    def get_skill_info(self, name: str) -> dict[str, Any] | None:
        """
        Get detailed information about a skill.
        
        Args:
            name: Name of the skill
            
        Returns:
            Dict with skill information or None if not found
        """
        skill_class = self._skills.get(name)
        if skill_class is None:
            return None
        
        return {
            "name": skill_class.name,
            "description": skill_class.description,
            "version": skill_class.version,
            "tags": skill_class.tags,
            "required_tools": skill_class.required_tools,
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.type,
                    "required": p.required,
                    "default": p.default,
                }
                for p in skill_class.parameters
            ],
        }
    
    def load_from_directory(self, directory: str) -> int:
        """
        Load skills from a directory.
        
        Looks for Python files with skill classes and registers them.
        
        Args:
            directory: Path to the directory containing skill files
            
        Returns:
            Number of skills loaded
        """
        loaded = 0
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return 0
        
        for file_path in dir_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            try:
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(
                    module_name, file_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, Skill)
                            and attr is not Skill
                        ):
                            self.register(attr)
                            loaded += 1
            except Exception as e:
                print(f"Error loading skill from {file_path}: {e}")
        
        return loaded


_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    """Get the global skill registry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def register_skill(skill_class: Type[Skill]) -> Type[Skill]:
    """
    Decorator to register a skill class.
    
    Usage:
        @register_skill
        class MySkill(Skill):
            name = "my_skill"
            ...
    """
    get_registry().register(skill_class)
    return skill_class
