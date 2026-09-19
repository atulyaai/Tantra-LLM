"""Tantra/ui.py — Console banner, status dashboard, and expert panel rendering.

Extracted from main.py (previously print_banner, print_status_dashboard,
print_expert_panel). Pure presentation functions: no side effects beyond
writing to stdout, no state owned here. Falls back to plain print() when
`rich` isn't installed, exactly as main.py did before this extraction.
"""
from __future__ import annotations

import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


def print_banner() -> None:
    is_tty = getattr(sys.stdout, "isatty", lambda: False)()
    if console and is_tty:
        try:
            console.print(Panel.fit(
                "  [bold cyan]तन्त्र[/bold cyan]  [bold]TANTRA LLM[/bold]\n"
                "  [dim]NeuroCore Architecture • CPU-First • Local AI[/dim]",
                title="[yellow]Initializing[/yellow]",
                border_style="cyan"
            ))
            return
        except Exception:
            pass
    print("")
    print("  तन्त्र  TANTRA LLM                                ")
    print("  NeuroCore Architecture • CPU-First • Local AI      ")
    print("")


def print_status_dashboard(model, trainer, expert_reg, rt) -> None:
    if console:
        table = Table(title="Tantra-LLM Status Dashboard", show_header=False)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        total_params = sum(p.numel() for p in model.parameters())
        if trainer.step_count > 0 and not trainer._is_resume:
            status = "FRESH_START (step 0)"
        elif trainer._is_resume and trainer.step_count == 0:
            status = "RESUMING (step 0)"
        elif trainer._is_resume:
            status = f"RESUMING (Step {trainer.step_count})"
        else:
            status = f"FRESH (Step {trainer.step_count})"

        table.add_row("Model", f"NeuroCore ({total_params/1e6:.1f}M params)")
        table.add_row("Device", f"{rt.device} | dtype: {rt.dtype}")
        table.add_row("Training Status", status)
        table.add_row("Best Loss", f"{trainer.best_loss:.4f}")
        table.add_row("Total Tokens", f"{trainer.total_tokens:,}")
        table.add_row("Experts", f"{len(expert_reg)} registered")
        table.add_row("Hardware", f"{rt.offload_strategy} | Batch: {rt.batch_size}")

        console.print(table)
    else:
        print("== TANTRA-LLM STATUS ==")
        print(f"Status: {'FRESH' if trainer.step_count == 0 else 'RESUMING (Step ' + str(trainer.step_count) + ')'}")


def print_expert_panel(expert_reg) -> None:
    if console:
        table = Table(title="Expert Registry", show_header=True, header_style="bold magenta")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Specialization")
        table.add_column("Params")
        table.add_column("Usage")
        table.add_column("Status")
        table.add_column("DNA File")

        for e_id, e_info in expert_reg.experts.items():
            spec = e_info.get("specialization", "unknown")
            emoji = ""
            if "language" in spec:
                emoji = ""
            elif "code" in spec:
                emoji = ""
            elif "math" in spec:
                emoji = ""

            table.add_row(
                str(e_id),
                e_info.get("name", f"expert_{e_id}"),
                f"{emoji} {spec}",
                f"{e_info.get('param_count', 0)/1e6:.1f}M",
                str(e_info.get("usage_count", 0)),
                "ACTIVE",
                e_info.get("dna_path", "None")
            )
        console.print(table)
    else:
        print("== EXPERT REGISTRY ==")
        for e_id, e_info in expert_reg.experts.items():
            print(f"Expert {e_id}: {e_info}")
