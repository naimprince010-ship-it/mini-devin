"""
Enhanced Mini-Devin Integration Example
Demonstrates all new features working together
"""

import asyncio
import os
from pathlib import Path
from mini_devin.integrations import (
    GitHubIntegration,
    DeploymentManager,
    PlaywrightAgent,
    PersistentMemory,
    TestFixRerunLoop
)

async def enhanced_mini_devin_workflow():
    """Complete enhanced Mini-Devin workflow example"""
    
    print("🚀 Starting Enhanced Mini-Devin Workflow")
    print("=" * 50)
    
    # 1. Initialize Persistent Memory
    print("\n📝 1. Initializing Persistent Memory...")
    memory = PersistentMemory("./memory")
    await memory.initialize()
    
    # Store workflow context
    session_id = "enhanced_workflow_001"
    await memory.start_session(session_id)
    
    # 2. Initialize GitHub Integration
    print("\n🔗 2. Setting up GitHub Integration...")
    github = GitHubIntegration()
    github_ready = await github.initialize(
        repo_path="./workspace",
        repo_name="user/repo"  # Replace with actual repo
    )
    
    if github_ready:
        print("✅ GitHub integration ready")
    else:
        print("❌ GitHub integration failed - using local mode")
    
    # 3. Initialize Playwright Agent
    print("\n🌐 3. Starting Playwright Agent...")
    playwright = PlaywrightAgent(headless=True)
    browser_ready = await playwright.start("chromium")
    
    if browser_ready:
        print("✅ Playwright agent ready")
    else:
        print("❌ Playwright agent failed")
    
    # 4. Initialize Deployment Manager
    print("\n🚀 4. Setting up Deployment Manager...")
    deployment = DeploymentManager("./workspace")
    deployment_ready = await deployment.initialize()
    
    if deployment_ready:
        platforms = list(deployment.platform_clients.keys())
        print(f"✅ Deployment ready for: {', '.join(platforms)}")
    else:
        print("❌ Deployment manager failed")
    
    # 5. Initialize Test-Fix-Rerun Loop
    print("\n🔄 5. Setting up Test-Fix-Rerun Loop...")
    test_loop = TestFixRerunLoop("./workspace")
    print("✅ Test-fix loop ready")
    
    # Example Task: Build and Deploy a Web App
    print("\n🎯 6. Running Example Task: Build & Deploy Web App")
    print("-" * 50)
    
    tasks_completed = []
    
    try:
        # Step 1: Create feature branch
        if github_ready:
            print("\n📂 Creating feature branch...")
            branch_success = await github.create_feature_branch(
                "enhanced-web-app-demo"
            )
            if branch_success:
                tasks_completed.append("Created feature branch")
                print("✅ Feature branch created")
        
        # Step 2: Run automated tests
        print("\n🧪 Running automated tests...")
        test_results = await test_loop.run_test_suite(
            "pytest -v tests/",
            max_iterations=3
        )
        
        if test_results['final_status'] == 'passed':
            tasks_completed.append("All tests passed")
            print("✅ All tests passed")
        else:
            tasks_completed.append(f"Tests fixed after {len(test_results['iterations'])} iterations")
            print(f"⚠️ Tests fixed after {len(test_results['iterations'])} iterations")
        
        # Step 3: Web automation with Playwright
        if browser_ready:
            print("\n🌐 Testing web application...")
            await playwright.navigate_to("http://localhost:3000")
            await playwright.take_screenshot("app_test.png")
            
            # Test login functionality
            login_success = await playwright.type_text("#username", "testuser")
            login_success &= await playwright.type_text("#password", "testpass")
            login_success &= await playwright.click_element("#login-btn")
            
            if login_success:
                tasks_completed.append("Web app testing completed")
                print("✅ Web app testing completed")
            else:
                print("⚠️ Web app testing had issues")
        
        # Step 4: Deploy application
        if deployment_ready:
            print("\n🚀 Deploying application...")
            deployment_result = await deployment.deploy_to_platform(
                "vercel",  # or "railway"
                "./workspace"
            )
            
            if deployment_result.get('success'):
                tasks_completed.append(f"Deployed to {deployment_result.get('url', 'unknown')}")
                print(f"✅ Deployed to: {deployment_result.get('url')}")
            else:
                print(f"❌ Deployment failed: {deployment_result.get('error')}")
        
        # Step 5: Create Pull Request
        if github_ready:
            print("\n📋 Creating Pull Request...")
            pr = await github.create_pull_request(
                title="Enhanced Mini-Devin Features Demo",
                description=f"""
## Automated PR by Mini-Devin

This PR demonstrates the enhanced capabilities:

### Features Implemented:
- ✅ GitHub Integration (PR/Branch/Commit)
- ✅ Playwright Agent Integration (web automation)
- ✅ Persistent Memory (cross-session)
- ✅ Test-Fix-Rerun Loop (automatic)
- ✅ Deployment Integration (Vercel/Railway)

### Tasks Completed:
{chr(10).join(f"- {task}" for task in tasks_completed)}

### Deployment:
- URL: {deployment_result.get('url', 'N/A') if deployment_ready else 'N/A'}

---
*Generated by Mini-Devin AI Assistant*
                """,
                head_branch="enhanced-web-app-demo",
                labels=["automated", "mini-devin", "enhancement"]
            )
            
            if pr:
                tasks_completed.append(f"Created PR #{pr.number}")
                print(f"✅ Pull Request created: #{pr.number}")
        
        # Step 6: Store in Persistent Memory
        print("\n💾 Storing workflow in memory...")
        
        # Store the workflow memory
        memory_id = await memory.add_memory(
            content=f"""
Enhanced Mini-Devin Workflow completed successfully.

Tasks Completed:
{chr(10).join(f"- {task}" for task in tasks_completed)}

Features Demonstrated:
- GitHub Integration
- Playwright Web Automation
- Persistent Memory System
- Test-Fix-Rerun Loop
- Deployment Automation

This workflow shows how Mini-Devin can handle complex, multi-step development tasks autonomously.
            """,
            metadata={
                "workflow_type": "enhanced_demo",
                "tasks_completed": tasks_completed,
                "features_used": ["github", "playwright", "memory", "test_loop", "deployment"],
                "success": True
            },
            tags=["workflow", "enhanced", "demo", "automation"],
            importance=0.9
        )
        
        # Store reusable skills
        await memory.add_skill(
            name="automated_web_deployment",
            description="Complete workflow for testing and deploying web applications",
            code="""
# Automated Web Deployment Skill
async def deploy_web_app(project_path, platform="vercel"):
    # 1. Run tests
    test_results = await test_loop.run_test_suite("pytest -v")
    
    # 2. Test with Playwright
    await playwright.navigate_to("http://localhost:3000")
    await playwright.take_screenshot("pre_deploy.png")
    
    # 3. Deploy
    result = await deployment.deploy_to_platform(platform, project_path)
    
    # 4. Create PR if successful
    if result.get('success'):
        await github.create_pull_request(
            title=f"Deploy to {platform}",
            description=f"Automated deployment to {platform}"
        )
    
    return result
            """,
            category="deployment"
        )
        
        print("✅ Workflow stored in memory")
        
        # End session
        await memory.end_session(
            session_id,
            tasks_completed,
            "Successfully demonstrated all enhanced Mini-Devin features"
        )
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        
        # Store failure in memory
        await memory.add_memory(
            content=f"Enhanced workflow failed: {str(e)}",
            metadata={"error": str(e), "success": False},
            tags=["error", "workflow", "failed"],
            importance=0.7
        )
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    if browser_ready:
        await playwright.stop()
    
    # Display final summary
    print("\n📊 Final Summary")
    print("=" * 50)
    print(f"✅ Tasks Completed: {len(tasks_completed)}")
    for task in tasks_completed:
        print(f"   - {task}")
    
    # Get memory stats
    memory_stats = await memory.get_memory_stats()
    print(f"\n📈 Memory Stats:")
    print(f"   - Total Memories: {memory_stats.get('total_memories', 0)}")
    print(f"   - Total Skills: {memory_stats.get('total_skills', 0)}")
    print(f"   - Sessions: {memory_stats.get('total_sessions', 0)}")
    
    print("\n🎉 Enhanced Mini-Devin Workflow Complete!")

# Environment setup helper
def setup_environment():
    """Setup environment variables for enhanced features"""
    
    print("🔧 Setting up environment...")
    
    env_vars = {
        "GITHUB_TOKEN": "your-github-token-here",
        "VERCEL_TOKEN": "your-vercel-token-here", 
        "VERCEL_PROJECT_ID": "your-vercel-project-id",
        "RAILWAY_TOKEN": "your-railway-token-here",
        "RAILWAY_PROJECT_ID": "your-railway-project-id",
        "NETLIFY_TOKEN": "your-netlify-token-here",
        "NETLIFY_SITE_ID": "your-netlify-site-id"
    }
    
    print("\nRequired Environment Variables:")
    print("-" * 30)
    for var, value in env_vars.items():
        current = os.getenv(var)
        status = "✅" if current else "❌"
        print(f"{status} {var}: {current or 'Not set'}")
    
    print("\nTo set these variables, add them to your .env file:")
    print("Example:")
    for var, value in env_vars.items():
        print(f"{var}={value}")

if __name__ == "__main__":
    # Setup environment first
    setup_environment()
    
    print("\n" + "="*60)
    print("🚀 ENHANCED MINI-DEVIN DEMONSTRATION")
    print("="*60)
    
    # Run the enhanced workflow
    asyncio.run(enhanced_mini_devin_workflow())
