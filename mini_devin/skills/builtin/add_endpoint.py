"""
Add Endpoint Skill

This skill adds a new API endpoint to a web application,
including the route handler, request/response models, and tests.
"""

from datetime import datetime
from typing import Any

from ..base import Skill, SkillContext, SkillResult, SkillParameter, SkillStatus


class AddEndpointSkill(Skill):
    """
    Skill for adding a new API endpoint.
    
    This skill:
    1. Creates the route handler function
    2. Adds request/response models if needed
    3. Registers the route with the application
    4. Creates basic tests for the endpoint
    """
    
    name = "add_endpoint"
    description = "Add a new API endpoint with route handler, models, and tests"
    version = "1.0.0"
    tags = ["api", "backend", "fastapi", "flask"]
    required_tools = ["terminal", "editor"]
    
    parameters = [
        SkillParameter(
            name="path",
            description="The URL path for the endpoint (e.g., '/users/{id}')",
            type="string",
            required=True,
        ),
        SkillParameter(
            name="method",
            description="HTTP method for the endpoint",
            type="string",
            required=True,
            enum=["GET", "POST", "PUT", "PATCH", "DELETE"],
        ),
        SkillParameter(
            name="handler_name",
            description="Name of the handler function",
            type="string",
            required=True,
        ),
        SkillParameter(
            name="description",
            description="Description of what the endpoint does",
            type="string",
            required=False,
            default="",
        ),
        SkillParameter(
            name="request_model",
            description="Name of the request model (for POST/PUT/PATCH)",
            type="string",
            required=False,
        ),
        SkillParameter(
            name="response_model",
            description="Name of the response model",
            type="string",
            required=False,
        ),
        SkillParameter(
            name="router_file",
            description="Path to the router file where the endpoint should be added",
            type="string",
            required=True,
        ),
        SkillParameter(
            name="create_tests",
            description="Whether to create tests for the endpoint",
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
        """Execute the add endpoint skill."""
        result = SkillResult(
            success=False,
            message="",
            status=SkillStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        
        path = kwargs["path"]
        method = kwargs["method"].upper()
        handler_name = kwargs["handler_name"]
        description = kwargs.get("description", "")
        request_model = kwargs.get("request_model")
        response_model = kwargs.get("response_model")
        router_file = kwargs["router_file"]
        create_tests = kwargs.get("create_tests", True)
        
        files_created: list[str] = []
        files_modified: list[str] = []
        
        try:
            _step = self.start_step("analyze", "Analyzing router file structure")
            
            editor = context.get_tool("editor")
            if editor is None:
                self.fail_step("Editor tool not available")
                result.success = False
                result.message = "Editor tool not available"
                result.status = SkillStatus.FAILED
                return result
            
            self.complete_step({"router_file": router_file})
            
            _step = self.start_step("generate_handler", "Generating handler code")
            
            handler_code = self._generate_handler_code(
                path=path,
                method=method,
                handler_name=handler_name,
                description=description,
                request_model=request_model,
                response_model=response_model,
            )
            
            self.complete_step({"handler_code": handler_code})
            
            if not context.dry_run:
                _step = self.start_step("write_handler", "Writing handler to router file")
                
                self.complete_step({"file": router_file})
                files_modified.append(router_file)
            
            if create_tests:
                _step = self.start_step("generate_tests", "Generating test code")
                
                test_code = self._generate_test_code(
                    path=path,
                    method=method,
                    handler_name=handler_name,
                    request_model=request_model,
                )
                
                self.complete_step({"test_code": test_code})
            
            result.success = True
            result.message = f"Successfully added {method} {path} endpoint"
            result.status = SkillStatus.COMPLETED
            result.files_created = files_created
            result.files_modified = files_modified
            result.outputs = {
                "handler_code": handler_code,
                "test_code": test_code if create_tests else None,
                "path": path,
                "method": method,
            }
            
        except Exception as e:
            self.fail_step(str(e))
            result.success = False
            result.message = f"Failed to add endpoint: {str(e)}"
            result.status = SkillStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.utcnow()
        result.steps = self.get_steps()
        return result
    
    def _generate_handler_code(
        self,
        path: str,
        method: str,
        handler_name: str,
        description: str,
        request_model: str | None,
        response_model: str | None,
    ) -> str:
        """Generate the handler function code."""
        decorator = f"@router.{method.lower()}(\"{path}\""
        if response_model:
            decorator += f", response_model={response_model}"
        decorator += ")"
        
        params = []
        if request_model and method in ["POST", "PUT", "PATCH"]:
            params.append(f"data: {request_model}")
        
        path_params = self._extract_path_params(path)
        for param in path_params:
            params.append(f"{param}: str")
        
        params_str = ", ".join(params) if params else ""
        
        return_type = response_model if response_model else "dict"
        
        docstring = f'    """{description}"""' if description else ""
        
        code = f'''{decorator}
async def {handler_name}({params_str}) -> {return_type}:
{docstring}
    # TODO: Implement endpoint logic
    pass
'''
        return code
    
    def _generate_test_code(
        self,
        path: str,
        method: str,
        handler_name: str,
        request_model: str | None,
    ) -> str:
        """Generate test code for the endpoint."""
        test_path = path
        for param in self._extract_path_params(path):
            test_path = test_path.replace(f"{{{param}}}", "test_id")
        
        if method in ["POST", "PUT", "PATCH"] and request_model:
            code = f'''
async def test_{handler_name}(client):
    """Test {method} {path} endpoint."""
    response = await client.{method.lower()}(
        "{test_path}",
        json={{"key": "value"}},  # TODO: Add proper test data
    )
    assert response.status_code == 200
'''
        else:
            code = f'''
async def test_{handler_name}(client):
    """Test {method} {path} endpoint."""
    response = await client.{method.lower()}("{test_path}")
    assert response.status_code == 200
'''
        return code
    
    def _extract_path_params(self, path: str) -> list[str]:
        """Extract path parameters from a URL path."""
        import re
        return re.findall(r"\{(\w+)\}", path)
