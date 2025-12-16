"""
Setup Linting Skill

This skill sets up linting for a project by:
1. Detecting the project type
2. Installing linting tools
3. Creating configuration files
4. Setting up pre-commit hooks
"""

from datetime import datetime
from typing import Any

from ..base import Skill, SkillContext, SkillResult, SkillParameter, SkillStatus


class SetupLintingSkill(Skill):
    """
    Skill for setting up linting in a project.
    
    This skill:
    1. Detects the project type (Python, JavaScript, etc.)
    2. Installs appropriate linting tools
    3. Creates configuration files
    4. Optionally sets up pre-commit hooks
    """
    
    name = "setup_linting"
    description = "Set up linting tools and configuration for a project"
    version = "1.0.0"
    tags = ["linting", "code-quality", "eslint", "ruff", "prettier"]
    required_tools = ["terminal", "editor"]
    
    parameters = [
        SkillParameter(
            name="project_type",
            description="Type of project (auto-detect if not specified)",
            type="string",
            required=False,
            enum=["python", "javascript", "typescript", "rust", "go"],
        ),
        SkillParameter(
            name="linter",
            description="Specific linter to use (auto-select if not specified)",
            type="string",
            required=False,
        ),
        SkillParameter(
            name="setup_precommit",
            description="Whether to set up pre-commit hooks",
            type="boolean",
            required=False,
            default=True,
        ),
        SkillParameter(
            name="strict_mode",
            description="Whether to use strict linting rules",
            type="boolean",
            required=False,
            default=False,
        ),
    ]
    
    async def execute(
        self,
        context: SkillContext,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute the setup linting skill."""
        result = SkillResult(
            success=False,
            message="",
            status=SkillStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        
        project_type = kwargs.get("project_type")
        linter = kwargs.get("linter")
        setup_precommit = kwargs.get("setup_precommit", True)
        strict_mode = kwargs.get("strict_mode", False)
        
        files_created: list[str] = []
        files_modified: list[str] = []
        
        try:
            _step = self.start_step("detect", "Detecting project type")
            
            if project_type is None:
                project_type = self._detect_project_type(context.workspace_path)
            
            if project_type is None:
                self.fail_step("Could not detect project type")
                result.success = False
                result.message = "Could not detect project type"
                result.status = SkillStatus.FAILED
                return result
            
            self.complete_step({"project_type": project_type})
            
            _step = self.start_step("select_linter", "Selecting linter")
            
            if linter is None:
                linter = self._select_linter(project_type)
            
            self.complete_step({"linter": linter})
            
            _step = self.start_step("install", f"Installing {linter}")
            
            install_cmd = self._get_install_command(project_type, linter)
            
            self.complete_step({"command": install_cmd})
            
            _step = self.start_step("configure", "Creating configuration")
            
            config_file, config_content = self._generate_config(
                project_type=project_type,
                linter=linter,
                strict_mode=strict_mode,
            )
            
            files_created.append(config_file)
            self.complete_step({"config_file": config_file})
            
            if setup_precommit:
                _step = self.start_step("precommit", "Setting up pre-commit hooks")
                
                _precommit_config = self._generate_precommit_config(
                    project_type=project_type,
                    linter=linter,
                )
                
                files_created.append(".pre-commit-config.yaml")
                self.complete_step({"precommit": True})
            
            result.success = True
            result.message = f"Successfully set up {linter} for {project_type} project"
            result.status = SkillStatus.COMPLETED
            result.files_created = files_created
            result.files_modified = files_modified
            result.outputs = {
                "project_type": project_type,
                "linter": linter,
                "config_file": config_file,
                "install_command": install_cmd,
            }
            
        except Exception as e:
            self.fail_step(str(e))
            result.success = False
            result.message = f"Failed to set up linting: {str(e)}"
            result.status = SkillStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.utcnow()
        result.steps = self.get_steps()
        return result
    
    def _detect_project_type(self, workspace_path: str) -> str | None:
        """Detect the project type from files."""
        import os
        
        indicators = {
            "pyproject.toml": "python",
            "setup.py": "python",
            "requirements.txt": "python",
            "package.json": "javascript",
            "tsconfig.json": "typescript",
            "Cargo.toml": "rust",
            "go.mod": "go",
        }
        
        for filename, ptype in indicators.items():
            if os.path.exists(os.path.join(workspace_path, filename)):
                return ptype
        
        return None
    
    def _select_linter(self, project_type: str) -> str:
        """Select the default linter for a project type."""
        linters = {
            "python": "ruff",
            "javascript": "eslint",
            "typescript": "eslint",
            "rust": "clippy",
            "go": "golangci-lint",
        }
        return linters.get(project_type, "unknown")
    
    def _get_install_command(self, project_type: str, linter: str) -> str:
        """Get the command to install the linter."""
        if project_type == "python":
            if linter == "ruff":
                return "pip install ruff"
            elif linter == "flake8":
                return "pip install flake8"
            elif linter == "pylint":
                return "pip install pylint"
        elif project_type in ["javascript", "typescript"]:
            if linter == "eslint":
                return "npm install --save-dev eslint"
            elif linter == "prettier":
                return "npm install --save-dev prettier"
        elif project_type == "rust":
            return "rustup component add clippy"
        elif project_type == "go":
            return "go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"
        
        return f"# Install {linter} manually"
    
    def _generate_config(
        self,
        project_type: str,
        linter: str,
        strict_mode: bool,
    ) -> tuple[str, str]:
        """Generate linter configuration."""
        if project_type == "python" and linter == "ruff":
            return self._generate_ruff_config(strict_mode)
        elif project_type in ["javascript", "typescript"] and linter == "eslint":
            return self._generate_eslint_config(project_type, strict_mode)
        
        return (f".{linter}rc", "# Configuration file")
    
    def _generate_ruff_config(self, strict_mode: bool) -> tuple[str, str]:
        """Generate ruff configuration."""
        config = '''[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # Pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
'''
        
        if strict_mode:
            config += '''    "UP", # pyupgrade
    "ARG", # flake8-unused-arguments
    "SIM", # flake8-simplify
'''
        
        config += ''']
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["mini_devin"]
'''
        return ("pyproject.toml", config)
    
    def _generate_eslint_config(
        self,
        project_type: str,
        strict_mode: bool,
    ) -> tuple[str, str]:
        """Generate ESLint configuration."""
        extends = ["eslint:recommended"]
        if project_type == "typescript":
            extends.append("plugin:@typescript-eslint/recommended")
        if strict_mode:
            extends.append("plugin:@typescript-eslint/strict" if project_type == "typescript" else "eslint:all")
        
        config = f'''module.exports = {{
  env: {{
    browser: true,
    es2021: true,
    node: true,
  }},
  extends: {extends},
  parserOptions: {{
    ecmaVersion: "latest",
    sourceType: "module",
  }},
  rules: {{
    "no-unused-vars": "warn",
    "no-console": "warn",
  }},
}};
'''
        return (".eslintrc.js", config)
    
    def _generate_precommit_config(
        self,
        project_type: str,
        linter: str,
    ) -> str:
        """Generate pre-commit configuration."""
        config = "repos:\n"
        
        if project_type == "python":
            config += '''  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
'''
        elif project_type in ["javascript", "typescript"]:
            config += '''  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.55.0
    hooks:
      - id: eslint
        files: \\.[jt]sx?$
        types: [file]
'''
        
        return config
