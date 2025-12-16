"""
CLI for Mini-Devin

This module provides a command-line interface to run the Mini-Devin agent.
"""

import argparse
import asyncio
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .orchestrator.agent import create_agent

console = Console()


def print_banner():
    """Print the Mini-Devin banner."""
    banner = """
    ╔══════════════════════════════════════════╗
    ║           Mini-Devin                     ║
    ║   Autonomous AI Software Engineer        ║
    ╚══════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def run_command(args):
    """Run Mini-Devin on a task."""
    print_banner()
    
    # Resolve working directory
    work_dir = args.dir or os.getcwd()
    work_dir = os.path.abspath(work_dir)
    
    if not os.path.isdir(work_dir):
        console.print(f"[red]Error: Directory not found: {work_dir}[/red]")
        sys.exit(1)
    
    console.print(f"[bold]Task:[/bold] {args.task}")
    console.print(f"[bold]Working Directory:[/bold] {work_dir}")
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print()
    
    # Check for API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: No API key provided.[/red]")
        console.print("Set OPENAI_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    verbose = not args.quiet
    
    async def run_agent():
        agent = await create_agent(
            model=args.model,
            api_key=api_key,
            working_directory=work_dir,
            verbose=verbose,
        )
        agent.max_iterations = args.max_iterations
        
        result = await agent.run_simple(args.task)
        return result, agent
    
    try:
        result, agent = asyncio.run(run_agent())
        
        console.print()
        console.print(Panel(
            Markdown(result),
            title="[bold green]Result[/bold green]",
            border_style="green",
        ))
        
        # Print usage stats
        usage = agent.llm.get_usage_stats()
        console.print()
        console.print(f"[dim]Tokens used: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


def interactive_command(args):
    """Start an interactive session with Mini-Devin."""
    print_banner()
    
    work_dir = args.dir or os.getcwd()
    work_dir = os.path.abspath(work_dir)
    
    console.print(f"[bold]Working Directory:[/bold] {work_dir}")
    console.print(f"[bold]Model:[/bold] {args.model}")
    console.print()
    console.print("[dim]Type 'exit' or 'quit' to end the session.[/dim]")
    console.print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
        sys.exit(1)
    
    async def run_interactive():
        agent = await create_agent(
            model=args.model,
            api_key=api_key,
            working_directory=work_dir,
            verbose=True,
        )
        
        while True:
            try:
                console.print()
                task = console.input("[bold cyan]Task>[/bold cyan] ")
                
                if task.lower() in ("exit", "quit", "q"):
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if not task.strip():
                    continue
                
                result = await agent.run_simple(task)
                
                console.print()
                console.print(Panel(
                    Markdown(result),
                    title="[bold green]Result[/bold green]",
                    border_style="green",
                ))
                
                # Clear conversation for next task
                agent.llm.clear_conversation()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue
    
    try:
        asyncio.run(run_interactive())
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)


def version_command(args):
    """Show the Mini-Devin version."""
    from . import __version__
    console.print(f"Mini-Devin version {__version__}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="mini-devin",
        description="Mini-Devin: An autonomous AI software engineer agent",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run Mini-Devin on a task")
    run_parser.add_argument("task", help="The task description")
    run_parser.add_argument("-d", "--dir", help="Working directory for the agent")
    run_parser.add_argument("-m", "--model", default="gpt-4o", help="LLM model to use")
    run_parser.add_argument("--max-iterations", type=int, default=50, help="Maximum iterations")
    run_parser.add_argument("-q", "--quiet", action="store_true", help="Disable verbose output")
    run_parser.add_argument("--api-key", help="API key for the LLM provider")
    run_parser.set_defaults(func=run_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start an interactive session")
    interactive_parser.add_argument("-d", "--dir", help="Working directory for the agent")
    interactive_parser.add_argument("-m", "--model", default="gpt-4o", help="LLM model to use")
    interactive_parser.set_defaults(func=interactive_command)
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=version_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
