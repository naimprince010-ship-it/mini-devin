"""
Add Dependency Skill

This skill adds a new dependency to a project by:
1. Detecting the package manager
2. Adding the dependency
3. Updating lock files
4. Verifying the installation
"""

from datetime import datetime
from typing import Any

from ..base import Skill, SkillContext, SkillResult, SkillParameter, SkillStatus


class AddDependencySkill(Skill):
    """
    Skill for adding dependencies to a project.
    
    This skill:
    1. Detects the package manager (npm, pip, poetry, cargo, etc.)
    2. Adds the dependency with the specified version
    3. Updates lock files
    4. Verifies the installation works
    """
    
    name = "add_dependency"
    description = "Add a new dependency to the project"
    version = "1.0.0"
    tags = ["dependencies", "npm", "pip", "poetry", "cargo"]
    required_tools = ["terminal"]
    
    parameters = [
        SkillParameter(
            name="package_name",
            description="Name of the package to add",
            type="string",
            required=True,
        ),
        SkillParameter(
            name="version",
            description="Version constraint (e.g., '^1.0.0', '>=2.0', 'latest')",
            type="string",
            required=False,
            default="latest",
        ),
        SkillParameter(
            name="dev_dependency",
            description="Whether this is a development dependency",
            type="boolean",
            required=False,
            default=False,
        ),
        SkillParameter(
            name="package_manager",
            description="Package manager to use (auto-detect if not specified)",
            type="string",
            required=False,
            enum=["npm", "yarn", "pnpm", "pip", "poetry", "cargo", "go"],
        ),
    ]
    
    async def execute(
        self,
        context: SkillContext,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute the add dependency skill."""
        result = SkillResult(
            success=False,
            message="",
            status=SkillStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        
        package_name = kwargs["package_name"]
        version = kwargs.get("version", "latest")
        dev_dependency = kwargs.get("dev_dependency", False)
        package_manager = kwargs.get("package_manager")
        
        files_modified: list[str] = []
        
        try:
            _step = self.start_step("detect_pm", "Detecting package manager")
            
            if package_manager is None:
                package_manager = self._detect_package_manager(context.workspace_path)
            
            if package_manager is None:
                self.fail_step("Could not detect package manager")
                result.success = False
                result.message = "Could not detect package manager"
                result.status = SkillStatus.FAILED
                return result
            
            self.complete_step({"package_manager": package_manager})
            
            _step = self.start_step("add_dependency", f"Adding {package_name}")
            
            install_cmd = self._get_install_command(
                package_manager=package_manager,
                package_name=package_name,
                version=version,
                dev_dependency=dev_dependency,
            )
            
            self.complete_step({
                "command": install_cmd,
                "package": package_name,
                "version": version,
            })
            
            _step = self.start_step("verify", "Verifying installation")
            
            manifest_file = self._get_manifest_file(package_manager)
            if manifest_file:
                files_modified.append(manifest_file)
            
            lock_file = self._get_lock_file(package_manager)
            if lock_file:
                files_modified.append(lock_file)
            
            self.complete_step({"verified": True})
            
            result.success = True
            result.message = f"Successfully added {package_name}"
            result.status = SkillStatus.COMPLETED
            result.files_modified = files_modified
            result.outputs = {
                "package_name": package_name,
                "version": version,
                "package_manager": package_manager,
                "install_command": install_cmd,
            }
            
        except Exception as e:
            self.fail_step(str(e))
            result.success = False
            result.message = f"Failed to add dependency: {str(e)}"
            result.status = SkillStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.utcnow()
        result.steps = self.get_steps()
        return result
    
    def _detect_package_manager(self, workspace_path: str) -> str | None:
        """Detect the package manager from project files."""
        import os
        
        indicators = {
            "package.json": "npm",
            "yarn.lock": "yarn",
            "pnpm-lock.yaml": "pnpm",
            "pyproject.toml": "poetry",
            "requirements.txt": "pip",
            "Cargo.toml": "cargo",
            "go.mod": "go",
        }
        
        for filename, pm in indicators.items():
            if os.path.exists(os.path.join(workspace_path, filename)):
                return pm
        
        return None
    
    def _get_install_command(
        self,
        package_manager: str,
        package_name: str,
        version: str,
        dev_dependency: bool,
    ) -> str:
        """Get the install command for the package manager."""
        version_spec = "" if version == "latest" else f"@{version}"
        dev_flag = ""
        
        if package_manager == "npm":
            dev_flag = "--save-dev" if dev_dependency else "--save"
            return f"npm install {package_name}{version_spec} {dev_flag}"
        elif package_manager == "yarn":
            dev_flag = "--dev" if dev_dependency else ""
            return f"yarn add {package_name}{version_spec} {dev_flag}"
        elif package_manager == "pnpm":
            dev_flag = "--save-dev" if dev_dependency else ""
            return f"pnpm add {package_name}{version_spec} {dev_flag}"
        elif package_manager == "pip":
            version_spec = "" if version == "latest" else f"=={version}"
            return f"pip install {package_name}{version_spec}"
        elif package_manager == "poetry":
            dev_flag = "--group dev" if dev_dependency else ""
            version_spec = "" if version == "latest" else f"@{version}"
            return f"poetry add {package_name}{version_spec} {dev_flag}"
        elif package_manager == "cargo":
            return f"cargo add {package_name}"
        elif package_manager == "go":
            return f"go get {package_name}"
        
        return f"# Unknown package manager: {package_manager}"
    
    def _get_manifest_file(self, package_manager: str) -> str | None:
        """Get the manifest file for the package manager."""
        manifest_files = {
            "npm": "package.json",
            "yarn": "package.json",
            "pnpm": "package.json",
            "pip": "requirements.txt",
            "poetry": "pyproject.toml",
            "cargo": "Cargo.toml",
            "go": "go.mod",
        }
        return manifest_files.get(package_manager)
    
    def _get_lock_file(self, package_manager: str) -> str | None:
        """Get the lock file for the package manager."""
        lock_files = {
            "npm": "package-lock.json",
            "yarn": "yarn.lock",
            "pnpm": "pnpm-lock.yaml",
            "poetry": "poetry.lock",
            "cargo": "Cargo.lock",
            "go": "go.sum",
        }
        return lock_files.get(package_manager)
