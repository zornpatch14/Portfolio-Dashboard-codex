from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from .config import MissingConfigurationError, get_settings
from .http_client import APIClient
from .options import build_selection_payload, downsample_option, params_option, selection_options


def _echo_json(payload: Any) -> None:
    click.echo(json.dumps(payload, indent=2, default=str))


def _prepare_client(base_url: Optional[str], token: Optional[str]) -> APIClient:
    try:
        settings = get_settings(base_url=base_url, token=token)
    except MissingConfigurationError as exc:
        raise click.ClickException(str(exc)) from exc
    return APIClient(settings=settings)


@click.group()
@click.option("--base-url", envvar="API_BASE_URL", help="API base URL (env: API_BASE_URL)")
@click.option("--token", envvar="API_TOKEN", help="API token for Authorization header")
@click.pass_context
def cli(ctx: click.Context, base_url: Optional[str], token: Optional[str]) -> None:
    """Portfolio Dashboard API command line wrapper."""

    ctx.obj = {"client": _prepare_client(base_url, token)}


@cli.command()
@click.argument("paths", type=click.Path(exists=True, dir_okay=False, path_type=Path), nargs=-1, required=True)
@click.pass_context
def ingest(ctx: click.Context, paths: tuple[Path, ...]) -> None:
    """Upload trade files to the API ingest endpoint."""

    client: APIClient = ctx.obj["client"]
    files = []
    handles = []
    try:
        for path in paths:
            handle = path.open("rb")
            handles.append(handle)
            files.append(("files", (path.name, handle, "application/octet-stream")))

        payload = client.post("/api/v1/upload", files=files)
        _echo_json(payload)
    finally:
        for handle in handles:
            handle.close()


@cli.command("list-files")
@click.pass_context
def list_files(ctx: click.Context) -> None:
    """List uploaded files."""

    client: APIClient = ctx.obj["client"]
    payload = client.get("/api/v1/files")
    _echo_json(payload)


@cli.command("selection-meta")
@click.pass_context
def selection_meta(ctx: click.Context) -> None:
    """Fetch selection helper metadata (symbols, intervals, strategies, dates)."""

    client: APIClient = ctx.obj["client"]
    payload = client.get("/api/v1/selection/meta")
    _echo_json(payload)


@cli.command()
@click.option(
    "--kind",
    type=click.Choice(
        [
            "equity",
            "equity-percent",
            "drawdown",
            "intraday-dd",
            "netpos",
            "margin",
            "histogram",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="Series endpoint to call.",
)
@downsample_option
@selection_options
@click.pass_context
def series(ctx: click.Context, kind: str, **selection: Any) -> None:
    """Fetch time series (equity/drawdown/netpos/margin/histogram)."""

    client: APIClient = ctx.obj["client"]
    params = build_selection_payload(**selection)
    path = f"/api/v1/series/{kind.lower()}"
    payload = client.get(path, params=params)
    _echo_json(payload)


@cli.command()
@selection_options
@click.pass_context
def metrics(ctx: click.Context, **selection: Any) -> None:
    """Fetch metrics for the current selection."""

    client: APIClient = ctx.obj["client"]
    params = build_selection_payload(**selection)
    payload = client.get("/api/v1/metrics", params=params)
    _echo_json(payload)


@cli.command()
@click.option(
    "--mode",
    default="returns",
    show_default=True,
    type=click.Choice(["drawdown_pct", "returns", "pl", "slope"], case_sensitive=False),
)
@selection_options
@click.pass_context
def correlations(ctx: click.Context, mode: str, **selection: Any) -> None:
    """Fetch correlation matrices."""

    client: APIClient = ctx.obj["client"]
    params = build_selection_payload(mode=mode, **selection)
    payload = client.get("/api/v1/correlations", params=params)
    _echo_json(payload)


@cli.command()
@selection_options
@click.pass_context
def cta(ctx: click.Context, **selection: Any) -> None:
    """Fetch CTA tables for the selection."""

    client: APIClient = ctx.obj["client"]
    params = build_selection_payload(**selection)
    payload = client.get("/api/v1/cta", params=params)
    _echo_json(payload)


@cli.group()
@click.pass_context
def optimizer(ctx: click.Context) -> None:
    """Run optimizers with polling for completion."""

    if "client" not in ctx.obj:
        ctx.obj["client"] = _prepare_client(None, None)


def _poll_job(client: APIClient, job_id: str, interval: float, timeout: float) -> Dict[str, Any]:
    start = time.monotonic()
    last_status: Dict[str, Any] = {}
    while True:
        last_status = client.get(f"/api/v1/jobs/{job_id}")
        status = last_status.get("status") or last_status.get("state")
        click.echo(f"Job {job_id}: {status}")

        if status in {"completed", "failed", "error", "done"}:
            return last_status

        if time.monotonic() - start > timeout:
            raise click.ClickException("Timed out waiting for job completion")
        time.sleep(interval)


def _submit_optimizer(
    client: APIClient,
    endpoint: str,
    selection: Dict[str, Any],
    extra_params: Dict[str, Any],
    wait: bool,
    interval: float,
    timeout: float,
) -> None:
    body: Dict[str, Any] = {"selection": selection}
    if extra_params:
        body["params"] = extra_params

    response = client.post(endpoint, json_body=body)
    job_id = response.get("job_id") or response.get("id")
    if not job_id:
        _echo_json(response)
        return

    click.echo(f"Submitted job {job_id}")
    if not wait:
        return

    final_status = _poll_job(client, job_id, interval=interval, timeout=timeout)
    _echo_json(final_status)


@optimizer.command("allocator")
@params_option
@selection_options
@click.option("--interval", "interval_", default=2.0, show_default=True, help="Polling interval in seconds.")
@click.option("--timeout", default=300.0, show_default=True, help="Maximum seconds to wait for completion.")
@click.option("--wait/--no-wait", default=True, show_default=True, help="Poll for completion.")
@click.pass_context
def optimizer_allocator(
    ctx: click.Context,
    param: Dict[str, Any],
    interval_: float,
    timeout: float,
    wait: bool,
    **selection: Any,
) -> None:
    """Run the allocator optimizer."""

    client: APIClient = ctx.obj["client"]
    selection_payload = build_selection_payload(**selection)
    _submit_optimizer(
        client,
        endpoint="/api/v1/optimizer/allocator",
        selection=selection_payload,
        extra_params=param,
        wait=wait,
        interval=interval_,
        timeout=timeout,
    )


@optimizer.command("riskfolio")
@params_option
@selection_options
@click.option("--interval", "interval_", default=2.0, show_default=True, help="Polling interval in seconds.")
@click.option("--timeout", default=300.0, show_default=True, help="Maximum seconds to wait for completion.")
@click.option("--wait/--no-wait", default=True, show_default=True, help="Poll for completion.")
@click.pass_context
def optimizer_riskfolio(
    ctx: click.Context,
    param: Dict[str, Any],
    interval_: float,
    timeout: float,
    wait: bool,
    **selection: Any,
) -> None:
    """Run the riskfolio optimizer."""

    client: APIClient = ctx.obj["client"]
    selection_payload = build_selection_payload(**selection)
    _submit_optimizer(
        client,
        endpoint="/api/v1/optimizer/riskfolio",
        selection=selection_payload,
        extra_params=param,
        wait=wait,
        interval=interval_,
        timeout=timeout,
    )


@cli.command()
@click.option("--kind", required=True, type=click.Choice(["trades", "metrics"], case_sensitive=False))
@click.option("--format", "fmt", default="csv", show_default=True, type=click.Choice(["csv", "parquet"], case_sensitive=False))
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Optional output file path.")
@selection_options
@click.pass_context
def export(ctx: click.Context, kind: str, fmt: str, output: Optional[Path], **selection: Any) -> None:
    """Download exports as CSV or Parquet."""

    client: APIClient = ctx.obj["client"]
    params = build_selection_payload(format=fmt, **selection)
    path = f"/api/v1/export/{kind.lower()}"
    response = client.get(path, params=params, expect_json=False, stream=True)

    target_path = output or Path(f"{kind.lower()}.{fmt.lower()}")
    with target_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)
    click.echo(f"Saved to {target_path}")


def main(argv: Optional[list[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    cli.main(args=argv, prog_name=os.path.basename(sys.argv[0]))


if __name__ == "__main__":
    main()
