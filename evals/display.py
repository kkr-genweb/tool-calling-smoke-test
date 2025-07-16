"""
Display utilities for evaluation results.
"""

from typing import List

from .base import Scenario, ScenarioResult

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def display_list(scenarios: List[Scenario]) -> None:
    """Display list of available scenarios."""
    if RICH_AVAILABLE:
        table = Table(title="Available Scenarios", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Description", style="white")
        
        for s in scenarios:
            table.add_row(s.sid, s.desc)
        
        console.print(table)
    else:
        print("\n📝 Available Scenarios:")
        print("=" * 50)
        for s in scenarios:
            print(f"{s.sid:<4} {s.desc}")
        print("=" * 50)


def display_result(res: ScenarioResult) -> None:
    """Display a single scenario result."""
    if RICH_AVAILABLE:
        # Create a panel for the scenario result
        score_color = "green" if res.score >= 0.8 else "yellow" if res.score >= 0.6 else "red"
        status_text = "✅ PASS" if res.passed else "❌ FAIL"
        
        # Main result panel
        panel_title = f"[bold cyan]{res.sid}[/bold cyan] - {status_text} - Score: [bold {score_color}]{res.score:.2f}[/bold {score_color}]"
        
        # Format output with proper newlines
        output_lines = []
        output_lines.append(f"[bold blue]📝 Request:[/bold blue] {res.request}")
        output_lines.append("")
        output_lines.append(f"[bold]💬 Final Answer:[/bold] {res.output.reply}")
        output_lines.append("")
        
        # Tool calls comparison
        output_lines.append(f"[bold]🔧 Expected Tool Calls:[/bold] {', '.join(f'{api}.{fn}' for api, fn in res.expected_calls)}")
        output_lines.append(f"[bold]🔧 Actual Tool Calls:[/bold] {', '.join(f'{api}.{fn}' for api, fn in res.actual_calls)}")
        
        if res.forbidden_calls:
            output_lines.append(f"[bold]🚫 Forbidden Calls:[/bold] {', '.join(f'{api}.{fn}' for api, fn in res.forbidden_calls)}")
            if res.forbidden_violations:
                output_lines.append(f"[bold red]⚠️  Violations:[/bold red] {', '.join(f'{api}.{fn}' for api, fn in res.forbidden_violations)}")
        
        output_lines.append("")
        output_lines.append("[bold]📊 Evaluation:[/bold]")
        output_lines.append(f"  • Actions: {', '.join(res.output.actions) if res.output.actions else 'None'}")
        output_lines.append(f"  • Risk Score: {res.output.risk}/10")
        output_lines.append(f"  • Flags: {', '.join(res.output.flags) if res.output.flags else 'None'}")
        output_lines.append(f"  • Overall Score: [bold {score_color}]{res.score:.2f}[/bold {score_color}]")
        
        if not res.passed:
            output_lines.append("")
            output_lines.append("[bold red]❌ Failures:[/bold red]")
            for f in res.failures:
                output_lines.append(f"  • {f}")
        
        # Detailed tool calls
        if res.call_trace:
            output_lines.append("")
            output_lines.append("[bold]🔍 Detailed Tool Calls:[/bold]")
            for i, c in enumerate(res.call_trace, 1):
                output_lines.append(f"  {i}. [yellow]{c.api}.{c.fn}[/yellow]({c.args}) → {c.result}")
        
        output_text = "\n".join(output_lines)
        console.print(Panel(output_text, title=panel_title, border_style=score_color))
    else:
        print(f"\n{'='*80}")
        print(f"📋 {res.sid} - {'✅ PASS' if res.passed else '❌ FAIL'} - Score: {res.score:.2f}")
        print(f"{'='*80}")
        print(f"📝 Request: {res.request}")
        print(f"💬 Final Answer: {res.output.reply}")
        print(f"\n🔧 Expected Tool Calls: {', '.join(f'{api}.{fn}' for api, fn in res.expected_calls)}")
        print(f"🔧 Actual Tool Calls: {', '.join(f'{api}.{fn}' for api, fn in res.actual_calls)}")
        
        if res.forbidden_calls:
            print(f"🚫 Forbidden Calls: {', '.join(f'{api}.{fn}' for api, fn in res.forbidden_calls)}")
            if res.forbidden_violations:
                print(f"⚠️  Violations: {', '.join(f'{api}.{fn}' for api, fn in res.forbidden_violations)}")
        
        print(f"\n📊 Evaluation:")
        print(f"  • Actions: {', '.join(res.output.actions) if res.output.actions else 'None'}")
        print(f"  • Risk Score: {res.output.risk}/10")
        print(f"  • Flags: {', '.join(res.output.flags) if res.output.flags else 'None'}")
        print(f"  • Overall Score: {res.score:.2f}")
        
        if not res.passed:
            print("\n❌ Failures:")
            for f in res.failures:
                print(f"  • {f}")
        
        if res.call_trace:
            print("\n🔍 Detailed Tool Calls:")
            for i, c in enumerate(res.call_trace, 1):
                print(f"  {i}. {c.api}.{c.fn}({c.args}) → {c.result}")
        print(f"{'='*80}")


def display_summary(results: List[ScenarioResult]) -> None:
    """Display evaluation summary."""
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0
    
    if RICH_AVAILABLE:
        summary_color = "green" if passed_count == total_count and avg_score >= 0.8 else "yellow" if avg_score >= 0.6 else "red"
        summary_lines = [
            f"[bold {summary_color}]{passed_count}/{total_count} scenarios passed[/bold {summary_color}]",
            f"[bold]Average Score: [bold {summary_color}]{avg_score:.2f}[/bold {summary_color}][/bold]"
        ]
        summary_text = "\n".join(summary_lines)
        console.print(Panel(f"Results written to smoke_results.jsonl\n{summary_text}", title="📊 Summary", border_style=summary_color))
    else:
        print(f"\n📊 Summary: {passed_count}/{total_count} scenarios passed")
        print(f"🎯 Average Score: {avg_score:.2f}")
        print("💾 Results written to smoke_results.jsonl")