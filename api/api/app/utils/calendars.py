from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals


@lru_cache(maxsize=1)
def get_us_futures_calendar() -> xcals.ExchangeCalendar:
    """Return the canonical US futures exchange calendar (CME/NYMEX/CBOT)."""

    try:
        return xcals.get_calendar("us_futures")
    except Exception:
        # Fallback to CMES calendar name for older exchange_calendars installations.
        return xcals.get_calendar("CMES")


__all__ = ["get_us_futures_calendar"]
