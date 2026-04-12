"""
Deployment Integration for Plodder
Supports Vercel, Railway, and other deployment platforms
"""

import os
import re
import asyncio
import json
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)

class DeploymentManager:
    """Deployment manager for multiple platforms"""
    
    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.deployment_configs = {}
        self.platform_clients = {}
        
    async def initialize(self) -> bool:
        """Initialize deployment manager"""
        try:
            # Load deployment configurations
            await self._load_deployment_configs()
            
            # Initialize platform clients
            await self._initialize_platforms()
            
            logger.info("Deployment manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize deployment manager: {e}")
            return False
    
    async def _load_deployment_configs(self) -> None:
        """Load deployment configurations from environment or config files"""
        config_file = self.workspace_dir / "deployment.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.deployment_configs = json.load(f)
                logger.info("Loaded deployment configurations from file")
            except Exception as e:
                logger.error(f"Failed to load deployment config: {e}")
                self.deployment_configs = {}
        else:
            # Load from environment variables
            self.deployment_configs = {
                'vercel': {
                    'token': os.getenv('VERCEL_TOKEN'),
                    'team_id': os.getenv('VERCEL_TEAM_ID'),
                    'project_id': os.getenv('VERCEL_PROJECT_ID')
                },
                'railway': {
                    'token': os.getenv('RAILWAY_TOKEN'),
                    'project_id': os.getenv('RAILWAY_PROJECT_ID'),
                    'environment_id': os.getenv('RAILWAY_ENVIRONMENT_ID')
                },
                'netlify': {
                    'token': os.getenv('NETLIFY_TOKEN'),
                    'site_id': os.getenv('NETLIFY_SITE_ID')
                }
            }
    
    async def _initialize_platforms(self) -> None:
        """Initialize platform-specific clients"""
        try:
            # Initialize Vercel client
            if self.deployment_configs.get('vercel', {}).get('token'):
                self.platform_clients['vercel'] = VercelDeployment(
                    self.deployment_configs['vercel']
                )
            
            # Initialize Railway client
            if self.deployment_configs.get('railway', {}).get('token'):
                self.platform_clients['railway'] = RailwayDeployment(
                    self.deployment_configs['railway']
                )
            
            # Initialize Netlify client
            if self.deployment_configs.get('netlify', {}).get('token'):
                self.platform_clients['netlify'] = NetlifyDeployment(
                    self.deployment_configs['netlify']
                )
            
            logger.info(f"Initialized {len(self.platform_clients)} platform clients")
            
        except Exception as e:
            logger.error(f"Failed to initialize platforms: {e}")
    
    async def deploy_to_platform(
        self, 
        platform: str, 
        project_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deploy to a specific platform"""
        try:
            if platform not in self.platform_clients:
                return {
                    'success': False,
                    'error': f'Platform {platform} not configured'
                }
            
            client = self.platform_clients[platform]
            deployment_path = project_path or str(self.workspace_dir)
            
            logger.info(f"Starting deployment to {platform}")
            result = await client.deploy(deployment_path, config or {})
            
            logger.info(f"Deployment to {platform} completed: {result.get('success', False)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to deploy to {platform}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_deployment_status(self, platform: str, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            if platform not in self.platform_clients:
                return {'error': f'Platform {platform} not configured'}
            
            client = self.platform_clients[platform]
            status = await client.get_deployment_status(deployment_id)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {'error': str(e)}
    
    async def rollback_deployment(self, platform: str, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment"""
        try:
            if platform not in self.platform_clients:
                return {'success': False, 'error': f'Platform {platform} not configured'}
            
            client = self.platform_clients[platform]
            result = await client.rollback(deployment_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to rollback deployment: {e}")
            return {'success': False, 'error': str(e)}
    
    async def list_deployments(self, platform: str, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent deployments"""
        try:
            if platform not in self.platform_clients:
                return []
            
            client = self.platform_clients[platform]
            deployments = await client.list_deployments(limit)
            
            return deployments
            
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")
            return []
    
    async def setup_project(
        self, 
        platform: str, 
        project_name: str,
        project_type: str = "auto"
    ) -> Dict[str, Any]:
        """Setup a new project on the platform"""
        try:
            if platform not in self.platform_clients:
                return {'success': False, 'error': f'Platform {platform} not configured'}
            
            client = self.platform_clients[platform]
            result = await client.setup_project(project_name, project_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to setup project: {e}")
            return {'success': False, 'error': str(e)}

class VercelDeployment:
    """Vercel deployment client"""
    
    def __init__(self, config: Dict[str, Any]):
        self.token = config.get('token')
        self.team_id = config.get('team_id')
        self.project_id = config.get('project_id')
        
    async def deploy(self, project_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Vercel"""
        try:
            # Check if Vercel CLI is installed
            if not await self._check_vercel_cli():
                return {'success': False, 'error': 'Vercel CLI not installed'}
            
            # Setup Vercel authentication
            if not await self._setup_auth():
                return {'success': False, 'error': 'Vercel authentication failed'}
            
            # Deploy project
            deploy_cmd = ['vercel', '--prod']
            
            if self.project_id:
                deploy_cmd.extend(['--scope', self.project_id])
            
            if config.get('preview'):
                deploy_cmd = ['vercel']
            
            result = await self._run_command(deploy_cmd, cwd=project_path)
            
            if result['exit_code'] == 0:
                # Extract deployment URL from output
                url = self._extract_deployment_url(result['stdout'])
                
                return {
                    'success': True,
                    'url': url,
                    'deployment_id': self._extract_deployment_id(result['stdout']),
                    'logs': result['stdout']
                }
            else:
                return {
                    'success': False,
                    'error': result['stderr'],
                    'logs': result['stdout']
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get Vercel deployment status"""
        try:
            result = await self._run_command([
                'vercel', 'inspect', deployment_id
            ])
            
            if result['exit_code'] == 0:
                return self._parse_vercel_inspect(result['stdout'])
            else:
                return {'error': result['stderr']}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def rollback(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback Vercel deployment"""
        try:
            result = await self._run_command([
                'vercel', 'promote', deployment_id, '--scope', self.project_id
            ])
            
            return {
                'success': result['exit_code'] == 0,
                'logs': result['stdout'],
                'error': result['stderr'] if result['exit_code'] != 0 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def list_deployments(self, limit: int) -> List[Dict[str, Any]]:
        """List Vercel deployments"""
        try:
            result = await self._run_command(['vercel', 'list', '--limit', str(limit)])
            
            if result['exit_code'] == 0:
                return self._parse_vercel_list(result['stdout'])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to list Vercel deployments: {e}")
            return []
    
    async def setup_project(self, project_name: str, project_type: str) -> Dict[str, Any]:
        """Create (or resolve) a Vercel project via the REST API."""
        try:
            if not self.token:
                return {'success': False, 'error': 'VERCEL_TOKEN not configured'}

            import httpx

            slug = re.sub(r"[^a-z0-9-]+", "-", (project_name or "project").lower()).strip("-")[
                :100
            ] or "project"
            params: Dict[str, Any] = {}
            if self.team_id:
                params["teamId"] = self.team_id

            framework_map: Dict[str, str | None] = {
                "next": "nextjs",
                "nextjs": "nextjs",
                "react": "create-react-app",
                "vue": "vue",
                "nuxt": "nuxtjs",
                "svelte": "svelte",
                "remix": "remix",
                "astro": "astro",
                "static": None,
                "auto": None,
            }
            pt = (project_type or "auto").lower()
            fw = framework_map.get(pt)

            body: Dict[str, Any] = {"name": slug}
            if fw:
                body["framework"] = fw

            headers = {"Authorization": f"Bearer {self.token}"}

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    "https://api.vercel.com/v9/projects",
                    headers=headers,
                    params=params,
                    json=body,
                )
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "success": True,
                        "project_id": data.get("id"),
                        "name": data.get("name", slug),
                        "message": "Project created",
                    }
                if r.status_code in (409, 400):
                    r2 = await client.get(
                        f"https://api.vercel.com/v9/projects/{slug}",
                        headers=headers,
                        params=params,
                    )
                    if r2.status_code == 200:
                        data = r2.json()
                        return {
                            "success": True,
                            "project_id": data.get("id"),
                            "name": data.get("name", slug),
                            "message": "Project already exists; using existing",
                        }
                err_text = (
                    r.text[:800]
                    if r.text
                    else getattr(r, "reason_phrase", None) or str(r.status_code)
                )
                return {
                    "success": False,
                    "error": err_text,
                    "status_code": r.status_code,
                }

        except Exception as e:
            logger.error("Vercel setup_project failed: %s", e)
            return {'success': False, 'error': str(e)}
    
    async def _check_vercel_cli(self) -> bool:
        """Check if Vercel CLI is installed"""
        try:
            result = await self._run_command(['vercel', '--version'])
            return result['exit_code'] == 0
        except:
            return False
    
    async def _setup_auth(self) -> bool:
        """Setup Vercel authentication"""
        try:
            # Set token in environment
            os.environ['VERCEL_TOKEN'] = self.token
            return True
        except:
            return False
    
    async def _run_command(self, cmd: List[str], cwd: str = ".") -> Dict[str, Any]:
        """Run a command and capture output"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore')
            }
        except Exception as e:
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _extract_deployment_url(self, output: str) -> str:
        """Extract deployment URL from Vercel output"""
        import re
        url_pattern = r'https://[a-zA-Z0-9-]+\.vercel\.app'
        match = re.search(url_pattern, output)
        return match.group(0) if match else ''
    
    def _extract_deployment_id(self, output: str) -> str:
        """Extract deployment ID from Vercel output"""
        import re
        id_pattern = r'dpl_[a-zA-Z0-9]+'
        match = re.search(id_pattern, output)
        return match.group(0) if match else ''
    
    def _parse_vercel_inspect(self, output: str) -> Dict[str, Any]:
        """Parse Vercel inspect output"""
        # This is a simplified parser
        return {
            'status': 'ready',
            'url': self._extract_deployment_url(output),
            'created': 'recently'
        }
    
    def _parse_vercel_list(self, output: str) -> List[Dict[str, Any]]:
        """Parse Vercel list output"""
        # This is a simplified parser
        deployments = []
        lines = output.split('\n')
        for line in lines:
            if 'vercel.app' in line:
                url = self._extract_deployment_url(line)
                deployments.append({
                    'url': url,
                    'status': 'ready',
                    'created': 'recently'
                })
        return deployments

class RailwayDeployment:
    """Railway deployment client"""
    
    def __init__(self, config: Dict[str, Any]):
        self.token = config.get('token')
        self.project_id = config.get('project_id')
        self.environment_id = config.get('environment_id')
    
    async def deploy(self, project_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Railway"""
        try:
            # Check if Railway CLI is installed
            if not await self._check_railway_cli():
                return {'success': False, 'error': 'Railway CLI not installed'}
            
            # Login to Railway
            if not await self._login():
                return {'success': False, 'error': 'Railway login failed'}
            
            # Deploy project
            deploy_cmd = ['railway', 'up']
            
            result = await self._run_command(deploy_cmd, cwd=project_path)
            
            if result['exit_code'] == 0:
                url = self._extract_railway_url(result['stdout'])
                
                return {
                    'success': True,
                    'url': url,
                    'deployment_id': self._extract_deployment_id(result['stdout']),
                    'logs': result['stdout']
                }
            else:
                return {
                    'success': False,
                    'error': result['stderr'],
                    'logs': result['stdout']
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get Railway deployment status"""
        try:
            result = await self._run_command(['railway', 'status'])
            
            if result['exit_code'] == 0:
                return self._parse_railway_status(result['stdout'])
            else:
                return {'error': result['stderr']}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def rollback(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback Railway deployment"""
        try:
            # Railway rollback would involve redeploying a previous commit
            result = await self._run_command(['railway', 'rollback', deployment_id])
            
            return {
                'success': result['exit_code'] == 0,
                'logs': result['stdout'],
                'error': result['stderr'] if result['exit_code'] != 0 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def list_deployments(self, limit: int) -> List[Dict[str, Any]]:
        """List Railway deployments"""
        try:
            result = await self._run_command(['railway', 'logs', '--limit', str(limit)])
            
            if result['exit_code'] == 0:
                return self._parse_railway_logs(result['stdout'])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to list Railway deployments: {e}")
            return []
    
    async def setup_project(self, project_name: str, project_type: str) -> Dict[str, Any]:
        """Setup Railway project"""
        try:
            result = await self._run_command(['railway', 'init', '--name', project_name])
            
            return {
                'success': result['exit_code'] == 0,
                'project_id': f'railway-project-{project_name}',
                'logs': result['stdout']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _check_railway_cli(self) -> bool:
        """Check if Railway CLI is installed"""
        try:
            result = await self._run_command(['railway', '--version'])
            return result['exit_code'] == 0
        except:
            return False
    
    async def _login(self) -> bool:
        """Login to Railway"""
        try:
            result = await self._run_command(['railway', 'login', '--token', self.token])
            return result['exit_code'] == 0
        except:
            return False
    
    async def _run_command(self, cmd: List[str], cwd: str = ".") -> Dict[str, Any]:
        """Run a command and capture output"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore')
            }
        except Exception as e:
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _extract_railway_url(self, output: str) -> str:
        """Extract Railway URL from output"""
        import re
        url_pattern = r'https://[a-zA-Z0-9-]+\.railway\.app'
        match = re.search(url_pattern, output)
        return match.group(0) if match else ''
    
    def _extract_deployment_id(self, output: str) -> str:
        """Extract deployment ID from Railway output"""
        import re
        id_pattern = r'[a-f0-9-]{36}'  # UUID pattern
        match = re.search(id_pattern, output)
        return match.group(0) if match else ''
    
    def _parse_railway_status(self, output: str) -> Dict[str, Any]:
        """Parse Railway status output"""
        return {
            'status': 'running',
            'url': self._extract_railway_url(output),
            'service': 'web'
        }
    
    def _parse_railway_logs(self, output: str) -> List[Dict[str, Any]]:
        """Parse Railway logs output"""
        deployments = []
        lines = output.split('\n')
        for line in lines:
            if 'railway.app' in line:
                url = self._extract_railway_url(line)
                deployments.append({
                    'url': url,
                    'status': 'running',
                    'created': 'recently'
                })
        return deployments

class NetlifyDeployment:
    """Netlify deployment client"""
    
    def __init__(self, config: Dict[str, Any]):
        self.token = config.get('token')
        self.site_id = config.get('site_id')
    
    async def deploy(self, project_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Netlify"""
        try:
            # Check if Netlify CLI is installed
            if not await self._check_netlify_cli():
                return {'success': False, 'error': 'Netlify CLI not installed'}
            
            # Login to Netlify
            if not await self._login():
                return {'success': False, 'error': 'Netlify login failed'}
            
            # Deploy project
            deploy_cmd = ['netlify', 'deploy', '--prod', '--dir=dist']
            
            result = await self._run_command(deploy_cmd, cwd=project_path)
            
            if result['exit_code'] == 0:
                url = self._extract_netlify_url(result['stdout'])
                
                return {
                    'success': True,
                    'url': url,
                    'deployment_id': self._extract_deployment_id(result['stdout']),
                    'logs': result['stdout']
                }
            else:
                return {
                    'success': False,
                    'error': result['stderr'],
                    'logs': result['stdout']
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get Netlify deployment status"""
        try:
            result = await self._run_command(['netlify', 'status'])
            
            if result['exit_code'] == 0:
                return self._parse_netlify_status(result['stdout'])
            else:
                return {'error': result['stderr']}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def rollback(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback Netlify deployment"""
        try:
            result = await self._run_command(['netlify', 'rollback', deployment_id])
            
            return {
                'success': result['exit_code'] == 0,
                'logs': result['stdout'],
                'error': result['stderr'] if result['exit_code'] != 0 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def list_deployments(self, limit: int) -> List[Dict[str, Any]]:
        """List Netlify deployments"""
        try:
            result = await self._run_command(['netlify', 'deploys', '--limit', str(limit)])
            
            if result['exit_code'] == 0:
                return self._parse_netlify_deploys(result['stdout'])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to list Netlify deployments: {e}")
            return []
    
    async def setup_project(self, project_name: str, project_type: str) -> Dict[str, Any]:
        """Setup Netlify project"""
        try:
            result = await self._run_command(['netlify', 'init'])
            
            return {
                'success': result['exit_code'] == 0,
                'site_id': f'netlify-site-{project_name}',
                'logs': result['stdout']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _check_netlify_cli(self) -> bool:
        """Check if Netlify CLI is installed"""
        try:
            result = await self._run_command(['netlify', '--version'])
            return result['exit_code'] == 0
        except:
            return False
    
    async def _login(self) -> bool:
        """Login to Netlify"""
        try:
            result = await self._run_command(['netlify', 'login', '--token', self.token])
            return result['exit_code'] == 0
        except:
            return False
    
    async def _run_command(self, cmd: List[str], cwd: str = ".") -> Dict[str, Any]:
        """Run a command and capture output"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore')
            }
        except Exception as e:
            return {
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _extract_netlify_url(self, output: str) -> str:
        """Extract Netlify URL from output"""
        import re
        url_pattern = r'https://[a-zA-Z0-9-]+\.netlify\.app'
        match = re.search(url_pattern, output)
        return match.group(0) if match else ''
    
    def _extract_deployment_id(self, output: str) -> str:
        """Extract deployment ID from Netlify output"""
        import re
        id_pattern = r'[a-f0-9]{12}'  # Netlify deployment ID pattern
        match = re.search(id_pattern, output)
        return match.group(0) if match else ''
    
    def _parse_netlify_status(self, output: str) -> Dict[str, Any]:
        """Parse Netlify status output"""
        return {
            'status': 'published',
            'url': self._extract_netlify_url(output),
            'site': 'active'
        }
    
    def _parse_netlify_deploys(self, output: str) -> List[Dict[str, Any]]:
        """Parse Netlify deploys output"""
        deployments = []
        lines = output.split('\n')
        for line in lines:
            if 'netlify.app' in line:
                url = self._extract_netlify_url(line)
                deployments.append({
                    'url': url,
                    'status': 'published',
                    'created': 'recently'
                })
        return deployments

# Example usage
async def automated_deployment_workflow(
    workspace_dir: str,
    platform: str = "vercel",
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """Complete automated deployment workflow"""
    
    manager = DeploymentManager(workspace_dir)
    await manager.initialize()
    
    result = await manager.deploy_to_platform(platform, project_path)
    
    return result
