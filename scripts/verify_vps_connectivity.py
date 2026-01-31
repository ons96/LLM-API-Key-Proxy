#!/usr/bin/env python3
"""
VPS Connectivity & API Gateway Verification Script
Usage: python scripts/verify_vps_connectivity.py [VPS_IP] [--port 8001]

This script tests:
1. TCP Connectivity to the VPS port (default 8001)
2. HTTP Health Endpoint (/health)
3. Provider Status API (/api/providers/status)
4. Model List (/v1/models)
"""

import sys
import argparse
import socket
import time
import json
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()


def check_tcp_port(ip, port, timeout=3):
    """Check if a TCP port is open."""
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        start_time = time.time()
        result = conn.connect_ex((ip, port))
        elapsed = (time.time() - start_time) * 1000
        conn.close()
        return result == 0, elapsed
    except Exception:
        return False, 0


def test_endpoint(base_url, endpoint, description):
    """Test a specific HTTP endpoint."""
    url = f"{base_url}{endpoint}"
    try:
        start_time = time.time()
        response = requests.get(url, timeout=5)
        elapsed = (time.time() - start_time) * 1000

        status_color = "green" if response.status_code == 200 else "red"
        console.print(f"  Testing {description} ([dim]{endpoint}[/dim])... ", end="")

        if response.status_code == 200:
            console.print(f"[{status_color}]OK[/{status_color}] ({elapsed:.0f}ms)")
            return True, response.json()
        else:
            console.print(
                f"[{status_color}]FAILED[/{status_color}] (Status: {response.status_code})"
            )
            return False, None

    except ConnectionError:
        console.print(f"  Testing {description}... [red]CONNECTION REFUSED[/red]")
        return False, None
    except Timeout:
        console.print(f"  Testing {description}... [red]TIMEOUT[/red]")
        return False, None
    except Exception as e:
        console.print(f"  Testing {description}... [red]ERROR: {str(e)}[/red]")
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Verify VPS API Gateway Connectivity")
    parser.add_argument("ip", nargs="?", help="VPS IP Address")
    parser.add_argument(
        "--port", type=int, default=8001, help="Port number (default: 8001)"
    )
    args = parser.parse_args()

    # Title
    console.print(
        Panel.fit(
            "[bold cyan]VPS Connectivity & API Gateway Verifier[/bold cyan]",
            border_style="cyan",
        )
    )

    # Get IP if not provided
    vps_ip = args.ip
    if not vps_ip:
        vps_ip = console.input(
            "[bold yellow]Enter VPS IP Address:[/bold yellow] "
        ).strip()
        if not vps_ip:
            console.print("[red]Error: IP address is required.[/red]")
            sys.exit(1)

    base_url = f"http://{vps_ip}:{args.port}"
    console.print(f"\nTargeting: [bold blue]{base_url}[/bold blue]\n")

    # 1. TCP Connectivity Check
    console.print("[bold]1. TCP Connectivity Check[/bold]")
    is_open, latency = check_tcp_port(vps_ip, args.port)

    if is_open:
        console.print(f"  Port {args.port}: [green]OPEN[/green] ({latency:.0f}ms)")
    else:
        console.print(f"  Port {args.port}: [red]CLOSED or FILTERED[/red]")
        console.print(
            "\n[bold red]CRITICAL FAILURE:[/bold red] Cannot connect to port. Check:"
        )
        console.print("  - Oracle Cloud Security List (Ingress Rules)")
        console.print("  - VPS Firewall (ufw status)")
        console.print("  - Is the service actually running?")
        sys.exit(1)

    # 2. HTTP Endpoint Checks
    console.print("\n[bold]2. API Endpoint Health[/bold]")

    # Check Root
    ok_root, _ = test_endpoint(base_url, "/", "Root Endpoint")

    # Check Health
    # Note: trying both /health (common) and /api/providers/health (from status_api.py)
    ok_health, _ = test_endpoint(
        base_url, "/api/providers/health", "Status Tracker Health"
    )

    # Check Provider Status
    ok_status, status_data = test_endpoint(
        base_url, "/api/providers/status", "Provider Status"
    )

    # Check Models
    ok_models, models_data = test_endpoint(base_url, "/v1/models", "Models List")

    # 3. Dashboard Data Summary
    if ok_status and status_data:
        console.print("\n[bold]3. Dashboard Status Summary[/bold]")

        providers = status_data.get("providers", {})

        table = Table(title=f"Active Providers (Total: {len(providers)})")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Latency", justify="right")
        table.add_column("Failures", justify="right")
        table.add_column("Last Check", style="dim")

        for name, p_data in providers.items():
            status = p_data.get("status", "unknown")
            status_style = (
                "green"
                if status == "healthy"
                else "yellow"
                if status == "degraded"
                else "red"
            )

            latency = p_data.get("latency_ms")
            latency_str = f"{latency:.0f}ms" if latency is not None else "-"

            table.add_row(
                name,
                f"[{status_style}]{status}[/{status_style}]",
                latency_str,
                str(p_data.get("consecutive_failures", 0)),
                p_data.get("last_check", "never"),
            )

        console.print(table)

    # 4. Final Verdict
    console.print("\n[bold]4. Verification Verdict[/bold]")
    if is_open and ok_status and ok_models:
        console.print(
            Panel(
                "[bold green]✅ SYSTEM OPERATIONAL[/bold green]\nVPS is accessible and API Gateway is functioning correctly.",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold yellow]⚠️  SYSTEM DEGRADED[/bold yellow]\nVPS is accessible but some endpoints are failing.",
                border_style="yellow",
            )
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Verification cancelled.[/yellow]")
