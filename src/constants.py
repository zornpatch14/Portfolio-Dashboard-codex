# src/constants.py

from __future__ import annotations
import re

APP_TITLE = "Portfolio Trade Analysis"
COLOR_PORTFOLIO = "black"
SHEET_HINTS = ["Trades List", "Trades", "Sheet1"]
DEFAULT_INITIAL_CAPITAL = 25_000.0

# Initial & maintenance margin per CONTRACT and Big Point Value
# NOTE: values are placeholders — keep your current numbers.
MARGIN_SPEC = {
    "MNQ": (3250.0, 3250.0, 2.0),
    "MES": (2300.0, 2300.0, 5.0),
    "MYM": (1500.0, 1500.0, 0.5),
    "M2K": (1000.0, 1000.0, 5.0),
    "CD":  (1100.0, 1100.0, 10.0),
    "JY":  (3100.0, 3100.0, 6.25),
    "NE1": (1500.0, 1500.0, 10.0),
    "NG":  (3800.0, 3800.0, 10.0),
}

# Conservative headroom for margin use (e.g., 1.00 = 100%)
MARGIN_BUFFER = 1.00

# Filename parsing regex (tradeslist_MNQ_15_3X.xlsx)
FILENAME_RE = re.compile(
    r"(?i)tradeslist[_-](?P<symbol>[A-Za-z]+)[_-](?P<interval>\d+)[_-](?P<strategy>[^.]+)"
)


POINT_VALUE = {
    "MNQ": 2.0,
    "MES": 5.0,
    "MYM": 1.0,
    "M2K": 5.0,
    "CD": 100000.0,
    "JY": 125000.0,
    "NE1": 100000.0,
    "NG": 10000.0,
}

