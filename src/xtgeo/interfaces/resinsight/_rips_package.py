"""Load the optional rips package and expose type aliases.

The ResInsight interface requires a minimum ``rips`` version that provides the
complete API surface used by xtgeo.  Rather than conditionally importing
individual symbols to support older releases, we enforce an all-or-nothing
version gate: either the installed ``rips`` meets :data:`MIN_RIPS_VERSION` and
every import succeeds, or the user is told to upgrade.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypeAlias

from packaging.version import InvalidVersion, Version

# Minimum rips version that exposes the full API used by xtgeo
MIN_RIPS_VERSION = "2026.2"


def _check_rips_version() -> None:
    """Raise ``RuntimeError`` if the installed rips version is too old or unparseable.

    Called at runtime (from :func:`require_rips`) rather than at import time so
    that the module can be loaded and inspected even when the installed ``rips``
    version does not meet the minimum requirement.
    """
    try:
        installed = version("rips")
    except PackageNotFoundError:
        raise RuntimeError(
            "rips package is installed but its version metadata is missing. "
            f"Please reinstall: pip install 'rips>={MIN_RIPS_VERSION}'"
        ) from None

    try:
        installed_ver = Version(installed)
    except InvalidVersion:
        raise RuntimeError(
            f"rips package reports version '{installed}' which is not a valid "
            f"PEP 440 version string. Please install a supported release: "
            f"pip install 'rips>={MIN_RIPS_VERSION}'"
        ) from None

    if installed_ver < Version(MIN_RIPS_VERSION):
        raise RuntimeError(
            f"xtgeo requires rips >= {MIN_RIPS_VERSION}, "
            f"but {installed} is installed. "
            f"Please upgrade: pip install 'rips>={MIN_RIPS_VERSION}'"
        )


def _load_package(package_name: str) -> Any | None:
    """Load a Python package by name, return ``None`` if unavailable."""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


def _import_rips_symbols(
    rips_module: Any,
) -> tuple[tuple[Any, Any, Any] | None, str | None]:
    """Extract the required API symbols from a loaded ``rips`` module.

    Returns a ``(symbols, error)`` pair. On success ``symbols`` is the
    ``(Case, Instance, Project)`` tuple and ``error`` is ``None``. On failure
    ``symbols`` is ``None`` and ``error`` is a user-facing message describing
    which part of the contract is missing.
    """
    try:
        return (
            rips_module.Case,
            rips_module.Instance,
            rips_module.Project,
        ), None
    except AttributeError as err:
        return None, (
            f"The installed rips package does not provide the required API "
            f"symbols (Case, Instance, Project): {err}. "
            f"Please upgrade: pip install 'rips>={MIN_RIPS_VERSION}'"
        )


def _resolve_rips() -> tuple[Any | None, str | None, tuple[Any, Any, Any]]:
    """Load ``rips`` and resolve the required API symbols.

    Returns ``(rips_module, error, (Case, Instance, Project))``. When ``rips``
    is unavailable or incomplete, ``rips_module`` is ``None`` and the symbol
    triple falls back to :data:`typing.Any` placeholders so the module can
    still be imported and inspected.
    """
    module = _load_package("rips")
    if module is None:
        return None, None, (Any, Any, Any)

    symbols, error = _import_rips_symbols(module)
    if symbols is None:
        return None, error, (Any, Any, Any)

    return module, None, symbols


rips, _rips_import_error, (_RipsCase, _RipsInstance, _RipsProject) = _resolve_rips()

RipsCaseType: TypeAlias = _RipsCase  # type: ignore[misc]
RipsInstanceType: TypeAlias = _RipsInstance  # type: ignore[misc]
RipsProjectType: TypeAlias = _RipsProject  # type: ignore[misc]

ResInsightInstanceOrPortType: TypeAlias = int | RipsInstanceType


def require_rips() -> None:
    """Raise ``RuntimeError`` if the ``rips`` package is not installed or too old.

    Call this at the top of any function that requires the rips package
    to provide a clear, consistent error message across all ResInsight
    interfaces.
    """
    if rips is None:
        raise RuntimeError(
            _rips_import_error
            or (
                "rips package is not available. Please install "
                f"rips >= {MIN_RIPS_VERSION} to use ResInsight features."
            )
        )
    _check_rips_version()
