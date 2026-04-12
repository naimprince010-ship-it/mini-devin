"""
Test-Fix-Rerun Loop for Plodder
Automatic testing, bug detection, and fixing cycle
"""

import os
import asyncio
import subprocess
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TestFixRerunLoop:
    """Automatic test-fix-rerun loop for continuous improvement"""
    
    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.test_results = {}
        self.fix_attempts = {}
        self.max_attempts = 3
        self.test_timeout = 30  # seconds
        
    async def run_test_suite(
        self, 
        test_command: str,
        test_files: Optional[List[str]] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run test suite and capture results"""
        try:
            start_time = datetime.now()
            
            # Prepare command
            if test_files:
                test_command = f"{test_command} {' '.join(test_files)}"
            
            # Run tests
            result = await self._run_command(
                test_command,
                cwd=self.workspace_dir,
                env=environment,
                timeout=self.test_timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            test_results = self._parse_test_output(result['stdout'], result['stderr'])
            test_results.update({
                'command': test_command,
                'exit_code': result['exit_code'],
                'duration': duration,
                'timestamp': start_time.isoformat(),
                'stdout': result['stdout'],
                'stderr': result['stderr']
            })
            
            logger.info(f"Test suite completed: {test_results.get('passed', 0)} passed, {test_results.get('failed', 0)} failed")
            return test_results
            
        except Exception as e:
            logger.error(f"Failed to run test suite: {e}")
            return {
                'error': str(e),
                'passed': 0,
                'failed': 0,
                'total': 0
            }
    
    async def detect_failing_tests(self, test_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect failing tests from test results"""
        failing_tests = []
        
        if 'test_cases' in test_results:
            for test_case in test_results['test_cases']:
                if test_case.get('status') == 'failed':
                    failing_tests.append(test_case)
        
        # Also parse from stderr for common test frameworks
        if test_results.get('stderr'):
            failing_tests.extend(self._parse_failing_tests_from_stderr(test_results['stderr']))
        
        return failing_tests
    
    async def analyze_failure(self, failing_test: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a failing test to determine root cause"""
        try:
            error_message = failing_test.get('error', '')
            test_name = failing_test.get('name', '')
            test_file = failing_test.get('file', '')
            
            analysis = {
                'test_name': test_name,
                'test_file': test_file,
                'error_message': error_message,
                'failure_type': self._classify_failure(error_message),
                'suggested_fixes': self._suggest_fixes(error_message, test_file),
                'affected_files': self._find_affected_files(error_message, test_file)
            }
            
            logger.info(f"Analyzed failure for {test_name}: {analysis['failure_type']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze failure: {e}")
            return {'error': str(e)}
    
    async def attempt_fix(
        self, 
        failure_analysis: Dict[str, Any],
        fix_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Attempt to fix the identified issue"""
        try:
            fix_type = failure_analysis.get('failure_type', 'unknown')
            affected_files = failure_analysis.get('affected_files', [])
            
            fix_result = {
                'fix_type': fix_type,
                'files_modified': [],
                'fix_applied': False,
                'fix_description': '',
                'confidence': 0.0
            }
            
            # Apply fix based on failure type
            if fix_type == 'import_error':
                fix_result = await self._fix_import_error(failure_analysis)
            elif fix_type == 'syntax_error':
                fix_result = await self._fix_syntax_error(failure_analysis)
            elif fix_type == 'assertion_error':
                fix_result = await self._fix_assertion_error(failure_analysis)
            elif fix_type == 'type_error':
                fix_result = await self._fix_type_error(failure_analysis)
            elif fix_type == 'null_pointer':
                fix_result = await self._fix_null_pointer(failure_analysis)
            elif fix_type == 'logic_error':
                fix_result = await self._fix_logic_error(failure_analysis)
            else:
                fix_result = await self._fix_generic_error(failure_analysis)
            
            logger.info(f"Fix attempt completed: {fix_result['fix_applied']}")
            return fix_result
            
        except Exception as e:
            logger.error(f"Failed to attempt fix: {e}")
            return {'error': str(e), 'fix_applied': False}
    
    async def run_fix_loop(
        self, 
        test_command: str,
        max_iterations: int = 3,
        stop_on_first_success: bool = True
    ) -> Dict[str, Any]:
        """Run the complete test-fix-rerun loop"""
        loop_results = {
            'iterations': [],
            'final_status': 'failed',
            'total_time': 0,
            'fixes_applied': 0
        }
        
        start_time = datetime.now()
        
        for iteration in range(max_iterations):
            logger.info(f"Starting test-fix iteration {iteration + 1}/{max_iterations}")
            
            # Run tests
            test_results = await self.run_test_suite(test_command)
            
            # Check if tests pass
            if test_results.get('failed', 0) == 0:
                logger.info("All tests passed!")
                loop_results['final_status'] = 'passed'
                break
            
            # Detect failing tests
            failing_tests = await self.detect_failing_tests(test_results)
            
            if not failing_tests:
                logger.warning("No failing tests detected but test suite failed")
                break
            
            # Analyze and fix each failing test
            iteration_fixes = 0
            for failing_test in failing_tests[:3]:  # Limit to 3 fixes per iteration
                analysis = await self.analyze_failure(failing_test)
                fix_result = await self.attempt_fix(analysis)
                
                if fix_result.get('fix_applied', False):
                    iteration_fixes += 1
                    loop_results['fixes_applied'] += 1
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'test_results': test_results,
                'failing_tests': len(failing_tests),
                'fixes_applied': iteration_fixes,
                'timestamp': datetime.now().isoformat()
            }
            loop_results['iterations'].append(iteration_result)
            
            # Stop if no fixes were applied
            if iteration_fixes == 0:
                logger.warning("No fixes applied in this iteration")
                break
            
            # Stop on first success if requested
            if stop_on_first_success and test_results.get('failed', 0) == 0:
                break
        
        end_time = datetime.now()
        loop_results['total_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"Test-fix loop completed: {loop_results['final_status']}")
        return loop_results
    
    async def _run_command(
        self, 
        command: str, 
        cwd: Path, 
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Run a command and capture output"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore')
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Command timed out: {command}")
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': 'Command timed out'
            }
        except Exception as e:
            logger.error(f"Failed to run command: {e}")
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _parse_test_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse test output from various test frameworks"""
        results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0,
            'test_cases': []
        }
        
        # Parse pytest output
        pytest_pattern = r'(\d+) passed, (\d+) failed, (\d+) skipped'
        pytest_match = re.search(pytest_pattern, stdout)
        if pytest_match:
            results['passed'] = int(pytest_match.group(1))
            results['failed'] = int(pytest_match.group(2))
            results['skipped'] = int(pytest_match.group(3))
            results['total'] = results['passed'] + results['failed'] + results['skipped']
        
        # Parse individual test failures
        failure_pattern = r'FAILED\s+(\S+)::(\S+)'
        for match in re.finditer(failure_pattern, stdout):
            test_case = {
                'name': match.group(2),
                'file': match.group(1),
                'status': 'failed',
                'error': ''
            }
            results['test_cases'].append(test_case)
        
        # Parse unittest output
        unittest_pattern = r'OK \(.*?\)|FAILED \(.*?\)'
        if 'OK' in stdout:
            results['passed'] = results.get('total', 1)
            results['failed'] = 0
        
        return results
    
    def _parse_failing_tests_from_stderr(self, stderr: str) -> List[Dict[str, Any]]:
        """Parse failing tests from stderr"""
        failing_tests = []
        
        # Common error patterns
        error_patterns = [
            r'File "(.*?)", line (\d+), in (\S+)',
            r'TestError: (.*?)',
            r'AssertionError: (.*?)',
            r'Exception: (.*?)'
        ]
        
        for pattern in error_patterns:
            for match in re.finditer(pattern, stderr):
                failing_tests.append({
                    'name': match.group(0),
                    'file': match.group(1) if len(match.groups()) > 0 else 'unknown',
                    'line': match.group(2) if len(match.groups()) > 1 else '0',
                    'error': match.group(0),
                    'status': 'failed'
                })
        
        return failing_tests
    
    def _classify_failure(self, error_message: str) -> str:
        """Classify the type of failure"""
        error_message = error_message.lower()
        
        if 'import' in error_message and 'not found' in error_message:
            return 'import_error'
        elif 'syntax' in error_message or 'invalid syntax' in error_message:
            return 'syntax_error'
        elif 'assertion' in error_message or 'assert' in error_message:
            return 'assertion_error'
        elif 'type' in error_message and 'error' in error_message:
            return 'type_error'
        elif 'none' in error_message or 'null' in error_message:
            return 'null_pointer'
        elif 'index' in error_message and ('out of range' in error_message or 'error' in error_message):
            return 'index_error'
        elif 'key' in error_message and ('not found' in error_message or 'error' in error_message):
            return 'key_error'
        else:
            return 'logic_error'
    
    def _suggest_fixes(self, error_message: str, test_file: str) -> List[str]:
        """Suggest potential fixes based on error message"""
        suggestions = []
        error_message = error_message.lower()
        
        if 'import' in error_message and 'not found' in error_message:
            suggestions.append("Check if the module is installed")
            suggestions.append("Verify the import path is correct")
            suggestions.append("Add missing module to requirements")
        
        elif 'syntax' in error_message:
            suggestions.append("Check for missing colons, brackets, or quotes")
            suggestions.append("Verify indentation is correct")
            suggestions.append("Check for unclosed strings or parentheses")
        
        elif 'assertion' in error_message:
            suggestions.append("Check the assertion logic")
            suggestions.append("Verify expected vs actual values")
            suggestions.append("Update test expectations if needed")
        
        elif 'type' in error_message:
            suggestions.append("Check variable types")
            suggestions.append("Add type conversion if needed")
            suggestions.append("Verify function return types")
        
        return suggestions
    
    def _find_affected_files(self, error_message: str, test_file: str) -> List[str]:
        """Find files that might be affected by the error"""
        files = [test_file]
        
        # Extract file paths from error message
        file_pattern = r'File "([^"]+)"'
        for match in re.finditer(file_pattern, error_message):
            file_path = match.group(1)
            if file_path not in files:
                files.append(file_path)
        
        return files
    
    async def _fix_import_error(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fix import errors"""
        try:
            error_message = analysis.get('error_message', '')
            affected_files = analysis.get('affected_files', [])
            
            # Extract missing module name
            import_pattern = r"No module named '([^']+)'"
            match = re.search(import_pattern, error_message)
            
            if match:
                missing_module = match.group(1)
                
                # Try to install the module
                install_result = await self._run_command(
                    f"pip install {missing_module}",
                    cwd=self.workspace_dir
                )
                
                if install_result['exit_code'] == 0:
                    return {
                        'fix_type': 'import_error',
                        'fix_applied': True,
                        'fix_description': f"Installed missing module: {missing_module}",
                        'files_modified': [],
                        'confidence': 0.8
                    }
            
            return {
                'fix_type': 'import_error',
                'fix_applied': False,
                'fix_description': 'Could not automatically fix import error',
                'files_modified': [],
                'confidence': 0.0
            }
            
        except Exception as e:
            return {'error': str(e), 'fix_applied': False}
    
    async def _fix_syntax_error(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fix syntax errors"""
        try:
            # Syntax errors usually need manual intervention
            # We can provide suggestions but automatic fixes are risky
            
            return {
                'fix_type': 'syntax_error',
                'fix_applied': False,
                'fix_description': 'Syntax errors require manual review',
                'files_modified': [],
                'confidence': 0.0
            }
            
        except Exception as e:
            return {'error': str(e), 'fix_applied': False}

    async def _fix_via_llm(self, analysis: Dict[str, Any], failure_label: str) -> Dict[str, Any]:
        """Apply bounded search/replace edits from the configured LLM (same helper as verification repair)."""
        from mini_devin.verification.llm_repair import run_llm_search_replace_repair

        err = analysis.get("error_message", "") or ""
        test_name = analysis.get("test_name", "") or ""
        test_file = analysis.get("test_file", "") or ""
        suggested = analysis.get("suggested_fixes") or []
        sug = "\n".join(suggested) if isinstance(suggested, list) else str(suggested)
        blob = "\n".join(
            [
                f"failure_type: {failure_label}",
                f"test_name: {test_name}",
                f"test_file: {test_file}",
                f"error_message:\n{err}",
                f"suggestions:\n{sug}",
            ]
        )
        instruction = (
            f"Fix this {failure_label.replace('_', ' ')} in the repository. "
            "Prefer fixing implementation code; change tests only if they are clearly wrong."
        )
        ok, msg, files = await run_llm_search_replace_repair(
            str(self.workspace_dir), instruction, blob
        )
        return {
            "fix_type": failure_label,
            "fix_applied": ok,
            "fix_description": msg,
            "files_modified": files,
            "confidence": 0.55 if ok else 0.0,
        }
    
    async def _fix_assertion_error(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fix assertion errors via LLM-assisted edits when configured."""
        try:
            return await self._fix_via_llm(analysis, "assertion_error")
        except Exception as e:
            return {'error': str(e), 'fix_applied': False}
    
    async def _fix_type_error(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fix type errors via LLM-assisted edits when configured."""
        try:
            return await self._fix_via_llm(analysis, "type_error")
        except Exception as e:
            return {'error': str(e), 'fix_applied': False}
    
    async def _fix_null_pointer(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fix null/None issues via LLM-assisted edits when configured."""
        try:
            return await self._fix_via_llm(analysis, "null_pointer")
        except Exception as e:
            return {'error': str(e), 'fix_applied': False}
    
    async def _fix_logic_error(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fix logic errors via LLM-assisted edits when configured."""
        try:
            return await self._fix_via_llm(analysis, "logic_error")
        except Exception as e:
            return {'error': str(e), 'fix_applied': False}
    
    async def _fix_generic_error(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Try LLM-assisted fix for uncategorized failures."""
        try:
            return await self._fix_via_llm(analysis, "generic_error")
        except Exception as e:
            return {'error': str(e), 'fix_applied': False}

# Example usage
async def automated_test_fix_workflow(
    workspace_dir: str,
    test_command: str = "pytest -v"
) -> Dict[str, Any]:
    """Complete automated test-fix workflow"""
    
    loop = TestFixRerunLoop(workspace_dir)
    results = await loop.run_fix_loop(test_command)
    
    return results
