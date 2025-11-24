from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import click


def _csv_to_list(_: click.Context, __: click.Parameter, value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def _iso_datetime(_: click.Context, __: click.Parameter, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.isoformat()
    except ValueError as exc:
        raise click.BadParameter("Use ISO 8601 date or datetime strings.") from exc


def _key_value_pair(_: click.Context, __: click.Parameter, value: tuple[str, ...]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for entry in value:
        if "=" not in entry:
            raise click.BadParameter("Parameters must be in key=value form.")
        key, val = entry.split("=", 1)
        params[key] = val
    return params


def selection_options(func: Callable[..., Any]) -> Callable[..., Any]:
    options = [
        click.option("--files", callback=_csv_to_list, help="Comma-separated file ids."),
        click.option("--symbols", callback=_csv_to_list, help="Comma-separated symbols filter."),
        click.option("--intervals", callback=_csv_to_list, help="Comma-separated intervals filter."),
        click.option("--strategies", callback=_csv_to_list, help="Comma-separated strategies filter."),
        click.option("--direction", type=click.Choice(["long", "short", "both"], case_sensitive=False)),
        click.option("--start-date", callback=_iso_datetime, help="ISO8601 start date/time."),
        click.option("--end-date", callback=_iso_datetime, help="ISO8601 end date/time."),
        click.option(
            "--contract-multipliers",
            callback=_csv_to_list,
            help="Optional contract multipliers (comma separated).",
        ),
        click.option(
            "--margin-overrides",
            callback=_csv_to_list,
            help="Optional margin overrides (comma separated).",
        ),
        click.option("--spike-flag", type=click.Choice(["on", "off", "auto"], case_sensitive=False)),
        click.option("--data-version", help="Data version identifier."),
    ]

    for option in reversed(options):
        func = option(func)
    return func


def downsample_option(func: Callable[..., Any]) -> Callable[..., Any]:
    return click.option("--downsample/--no-downsample", default=True, show_default=True)(func)


def params_option(func: Callable[..., Any]) -> Callable[..., Any]:
    return click.option(
        "--param",
        multiple=True,
        callback=_key_value_pair,
        help="Additional optimizer parameter in key=value form. Can be repeated.",
    )(func)


def build_selection_payload(**kwargs: Any) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}
