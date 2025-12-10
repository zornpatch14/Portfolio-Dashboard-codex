"""
Ensure the top-level ``api`` package exposes the application code that lives
under ``api/api`` so imports like ``api.app.ingest`` resolve whether or not the
project root is on ``PYTHONPATH``.
"""
from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_CURRENT_DIR = Path(__file__).resolve().parent
_INNER_PACKAGE = _CURRENT_DIR / "api"

if _INNER_PACKAGE.is_dir():
    # Append the inner package path so ``api.app`` resolves in editors/tests.
    __path__.append(str(_INNER_PACKAGE))
