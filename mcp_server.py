"""MCP Server for the xtgeo subsurface modelling library.

Exposes xtgeo's core functionality — surfaces, cubes, grids, wells,
points, and polygons — as MCP tools for LLM-driven workflows.
"""

from __future__ import annotations

import io
import json
import tempfile
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

import xtgeo

mcp = FastMCP(
    "xtgeo",
    instructions=(
        "MCP server for the xtgeo subsurface modelling library. "
        "Provides tools to load, inspect, manipulate, and export "
        "surfaces, seismic cubes, 3D grids, wells, points, and polygons."
    ),
)

# ---------------------------------------------------------------------------
# In-memory object store – keeps loaded xtgeo objects between tool calls.
# Each object gets a unique key returned to the caller.
# ---------------------------------------------------------------------------
_store: dict[str, Any] = {}
_counter: int = 0


def _put(obj: Any, prefix: str = "obj") -> str:
    global _counter
    _counter += 1
    key = f"{prefix}_{_counter}"
    _store[key] = obj
    return key


def _get(key: str) -> Any:
    if key not in _store:
        raise ValueError(
            f"Object '{key}' not found in store. "
            f"Available keys: {list(_store.keys())}"
        )
    return _store[key]


# ───────────────────────────── Store management ─────────────────────────────


@mcp.tool()
def list_objects() -> dict[str, str]:
    """List all objects currently held in the in-memory store.

    Returns a mapping of key → object type/name summary.
    """
    result = {}
    for key, obj in _store.items():
        name = getattr(obj, "name", "")
        result[key] = f"{type(obj).__name__}" + (f" ({name})" if name else "")
    return result


@mcp.tool()
def remove_object(key: str) -> str:
    """Remove an object from the in-memory store by its key."""
    if key not in _store:
        return f"Key '{key}' not found."
    del _store[key]
    return f"Removed '{key}'."


# ───────────────────────────── RegularSurface ───────────────────────────────


@mcp.tool()
def load_surface(filepath: str, fformat: str | None = None) -> dict:
    """Load a surface (map) from file.

    Supported formats: irap_binary, irap_ascii, ijxyz, petromod, zmap_ascii,
    xtg, hdf5 (auto-detected by default).

    Returns the store key and basic metadata.
    """
    filepath = os.path.expanduser(filepath)
    surf = xtgeo.surface_from_file(filepath, fformat=fformat)
    key = _put(surf, "surf")
    return {
        "key": key,
        "name": surf.name,
        "ncol": surf.ncol,
        "nrow": surf.nrow,
        "xori": surf.xori,
        "yori": surf.yori,
        "xinc": surf.xinc,
        "yinc": surf.yinc,
        "rotation": surf.rotation,
        "values_min": float(np.nanmin(surf.values)),
        "values_max": float(np.nanmax(surf.values)),
        "values_mean": float(np.nanmean(surf.values)),
    }


@mcp.tool()
def surface_info(key: str) -> dict:
    """Return metadata and statistics for a loaded surface."""
    surf = _get(key)
    vals = surf.values
    return {
        "name": surf.name,
        "ncol": surf.ncol,
        "nrow": surf.nrow,
        "xori": surf.xori,
        "yori": surf.yori,
        "xinc": surf.xinc,
        "yinc": surf.yinc,
        "rotation": surf.rotation,
        "values_min": float(np.nanmin(vals)),
        "values_max": float(np.nanmax(vals)),
        "values_mean": float(np.nanmean(vals)),
        "values_std": float(np.nanstd(vals)),
        "undefined_count": int(np.count_nonzero(np.isnan(vals))),
    }


@mcp.tool()
def save_surface(key: str, filepath: str, fformat: str | None = None) -> str:
    """Export a surface to file.

    Supported formats: irap_binary (default), irap_ascii, ijxyz, zmap_ascii,
    petromod, xtg, hdf5.
    """
    filepath = os.path.expanduser(filepath)
    surf = _get(key)
    surf.to_file(filepath, fformat=fformat)
    return f"Surface saved to {filepath}"


@mcp.tool()
def create_surface(
    ncol: int,
    nrow: int,
    xinc: float,
    yinc: float,
    xori: float = 0.0,
    yori: float = 0.0,
    rotation: float = 0.0,
    fill_value: float = 0.0,
) -> dict:
    """Create a new regular surface filled with a constant value."""
    values = np.full((ncol, nrow), fill_value, dtype=np.float64)
    surf = xtgeo.RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xinc=xinc,
        yinc=yinc,
        xori=xori,
        yori=yori,
        rotation=rotation,
        values=values,
    )
    key = _put(surf, "surf")
    return {"key": key, "ncol": ncol, "nrow": nrow}


@mcp.tool()
def surface_resample(source_key: str, target_key: str, sampling: str = "bilinear") -> str:
    """Resample *source* surface onto the geometry of *target* surface.

    The target surface values are overwritten with the resampled result.
    """
    source = _get(source_key)
    target = _get(target_key)
    target.resample(source, sampling=sampling)
    return f"Resampled {source_key} onto {target_key}."


@mcp.tool()
def surface_operation(key: str, operation: str, value: float | None = None) -> dict:
    """Apply a simple arithmetic or statistical operation to a surface.

    Supported operations: add, subtract, multiply, divide (requires value),
    smooth, fill (fill undefined nodes), copy (returns new key).
    """
    surf = _get(key)
    if operation == "add":
        surf.values += value
    elif operation == "subtract":
        surf.values -= value
    elif operation == "multiply":
        surf.values *= value
    elif operation == "divide":
        if value == 0:
            return {"error": "Cannot divide by zero."}
        surf.values /= value
    elif operation == "smooth":
        surf.smooth(iterations=1 if value is None else int(value))
    elif operation == "copy":
        new_key = _put(surf.copy(), "surf")
        return {"new_key": new_key}
    elif operation == "fill":
        surf.fill()
    else:
        return {"error": f"Unknown operation: {operation}"}
    return {
        "key": key,
        "values_min": float(np.nanmin(surf.values)),
        "values_max": float(np.nanmax(surf.values)),
        "values_mean": float(np.nanmean(surf.values)),
    }


@mcp.tool()
def surface_get_value_at_xy(key: str, x: float, y: float, sampling: str = "bilinear") -> dict:
    """Sample a surface value at a given (x, y) coordinate."""
    surf = _get(key)
    val = surf.get_value_from_xy(point=(x, y), sampling=sampling)
    return {"x": x, "y": y, "value": float(val) if val is not None else None}


@mcp.tool()
def surface_slice_cube(surface_key: str, cube_key: str, sampling: str = "nearest") -> dict:
    """Slice a seismic cube along a surface. Updates the surface values in-place."""
    surf = _get(surface_key)
    cube = _get(cube_key)
    surf.slice_cube(cube, sampling=sampling)
    return {
        "key": surface_key,
        "values_min": float(np.nanmin(surf.values)),
        "values_max": float(np.nanmax(surf.values)),
        "values_mean": float(np.nanmean(surf.values)),
    }


# ────────────────────────────── Seismic Cube ────────────────────────────────


@mcp.tool()
def load_cube(filepath: str, fformat: str = "guess") -> dict:
    """Load a seismic cube from file (SEGY, Storm, XTG)."""
    filepath = os.path.expanduser(filepath)
    cube = xtgeo.cube_from_file(filepath, fformat=fformat)
    key = _put(cube, "cube")
    return {
        "key": key,
        "ncol": cube.ncol,
        "nrow": cube.nrow,
        "nlay": cube.nlay,
        "xori": cube.xori,
        "yori": cube.yori,
        "zori": cube.zori,
        "xinc": cube.xinc,
        "yinc": cube.yinc,
        "zinc": cube.zinc,
        "rotation": cube.rotation,
        "values_min": float(np.nanmin(cube.values)),
        "values_max": float(np.nanmax(cube.values)),
    }


@mcp.tool()
def cube_info(key: str) -> dict:
    """Return metadata and statistics for a loaded cube."""
    cube = _get(key)
    return {
        "ncol": cube.ncol,
        "nrow": cube.nrow,
        "nlay": cube.nlay,
        "xori": cube.xori,
        "yori": cube.yori,
        "zori": cube.zori,
        "xinc": cube.xinc,
        "yinc": cube.yinc,
        "zinc": cube.zinc,
        "rotation": cube.rotation,
        "values_min": float(np.nanmin(cube.values)),
        "values_max": float(np.nanmax(cube.values)),
        "values_mean": float(np.nanmean(cube.values)),
    }


@mcp.tool()
def save_cube(key: str, filepath: str, fformat: str = "segy") -> str:
    """Export a cube to file (segy, storm, xtg)."""
    filepath = os.path.expanduser(filepath)
    cube = _get(key)
    cube.to_file(filepath, fformat=fformat)
    return f"Cube saved to {filepath}"


# ──────────────────────────────── 3D Grid ───────────────────────────────────


@mcp.tool()
def load_grid(filepath: str, fformat: str | None = None) -> dict:
    """Load a 3D grid from file.

    Supported formats: egrid, grdecl, roff, xtg (auto-detected by default).
    """
    filepath = os.path.expanduser(filepath)
    grid = xtgeo.grid_from_file(filepath, fformat=fformat)
    key = _put(grid, "grid")
    dims = grid.dimensions
    return {
        "key": key,
        "name": grid.name,
        "ncol": dims[0],
        "nrow": dims[1],
        "nlay": dims[2],
        "nactive": int(grid.nactive),
    }


@mcp.tool()
def grid_info(key: str) -> dict:
    """Return metadata for a loaded grid."""
    grid = _get(key)
    dims = grid.dimensions
    bbox = grid.get_bounding_box()
    return {
        "name": grid.name,
        "ncol": dims[0],
        "nrow": dims[1],
        "nlay": dims[2],
        "nactive": int(grid.nactive),
        "bounding_box": bbox,
        "subgrids": grid.get_subgrids(),
    }


@mcp.tool()
def save_grid(key: str, filepath: str, fformat: str | None = None) -> str:
    """Export a 3D grid to file (roff, grdecl, egrid, xtg)."""
    filepath = os.path.expanduser(filepath)
    grid = _get(key)
    grid.to_file(filepath, fformat=fformat)
    return f"Grid saved to {filepath}"


@mcp.tool()
def grid_get_dataframe(key: str, include_actnum: bool = True) -> dict:
    """Get the grid cell-centre coordinates and actnum as a JSON-friendly summary.

    Returns first/last rows and shape information (full dataframe may be huge).
    """
    grid = _get(key)
    df = grid.get_dataframe(activeonly=not include_actnum)
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "head": df.head(10).to_dict(orient="records"),
        "tail": df.tail(5).to_dict(orient="records"),
    }


# ──────────────────────────── Grid Properties ───────────────────────────────


@mcp.tool()
def load_grid_property(
    filepath: str,
    fformat: str | None = None,
    name: str | None = None,
    grid_key: str | None = None,
) -> dict:
    """Load a grid property from file.

    Optionally specify a grid key to associate geometry.
    Supported formats: roff, init, unrst, grdecl, bgrdecl, xtg.
    """
    filepath = os.path.expanduser(filepath)
    grid = _get(grid_key) if grid_key else None
    kwargs: dict[str, Any] = {"pfile": filepath}
    if fformat:
        kwargs["fformat"] = fformat
    if name:
        kwargs["name"] = name
    if grid:
        kwargs["grid"] = grid
    prop = xtgeo.gridproperty_from_file(**kwargs)
    key = _put(prop, "gprop")
    vals = prop.values
    return {
        "key": key,
        "name": prop.name,
        "dimensions": list(prop.dimensions),
        "values_min": float(np.nanmin(vals)),
        "values_max": float(np.nanmax(vals)),
        "values_mean": float(np.nanmean(vals)),
    }


@mcp.tool()
def save_grid_property(key: str, filepath: str, fformat: str = "roff") -> str:
    """Export a grid property to file."""
    filepath = os.path.expanduser(filepath)
    prop = _get(key)
    prop.to_file(filepath, fformat=fformat)
    return f"Grid property saved to {filepath}"


@mcp.tool()
def list_grid_properties_in_file(filepath: str, fformat: str | None = None) -> list[str]:
    """List property names available in a grid property file (roff, init, unrst)."""
    filepath = os.path.expanduser(filepath)
    kwargs: dict[str, Any] = {"property_file": filepath}
    if fformat:
        kwargs["fformat"] = fformat
    return xtgeo.list_gridproperties(**kwargs)


# ─────────────────────────────── Wells ──────────────────────────────────────


@mcp.tool()
def load_well(
    filepath: str,
    fformat: str | None = None,
    mdlogname: str | None = None,
    zonelogname: str | None = None,
) -> dict:
    """Load a well from file (RMS ASCII, CSV, HDF5)."""
    filepath = os.path.expanduser(filepath)
    kwargs: dict[str, Any] = {"wfile": filepath}
    if fformat:
        kwargs["fformat"] = fformat
    if mdlogname:
        kwargs["mdlogname"] = mdlogname
    if zonelogname:
        kwargs["zonelogname"] = zonelogname
    well = xtgeo.well_from_file(**kwargs)
    key = _put(well, "well")
    df = well.get_dataframe()
    return {
        "key": key,
        "name": well.name,
        "xpos": well.xpos,
        "ypos": well.ypos,
        "lognames": well.lognames,
        "nrows": len(df),
    }


@mcp.tool()
def well_info(key: str) -> dict:
    """Return metadata for a loaded well."""
    well = _get(key)
    df = well.get_dataframe()
    return {
        "name": well.name,
        "xpos": well.xpos,
        "ypos": well.ypos,
        "lognames": well.lognames,
        "nrows": len(df),
        "head": df.head(10).to_dict(orient="records"),
    }


@mcp.tool()
def well_get_dataframe(key: str, head_rows: int = 20) -> dict:
    """Get the well log dataframe summary."""
    well = _get(key)
    df = well.get_dataframe()
    return {
        "columns": list(df.columns),
        "shape": list(df.shape),
        "head": df.head(head_rows).to_dict(orient="records"),
        "describe": df.describe().to_dict(),
    }


@mcp.tool()
def save_well(key: str, filepath: str, fformat: str = "rms_ascii") -> str:
    """Export a well to file (rms_ascii, csv, hdf5)."""
    filepath = os.path.expanduser(filepath)
    well = _get(key)
    well.to_file(filepath, fformat=fformat)
    return f"Well saved to {filepath}"


@mcp.tool()
def well_get_surface_picks(well_key: str, surface_key: str) -> dict:
    """Get the intersection points between a well trajectory and a surface."""
    well = _get(well_key)
    surf = _get(surface_key)
    picks = well.get_surface_picks(surf)
    if picks is None:
        return {"picks": []}
    df = picks.get_dataframe()
    return {"picks": df.to_dict(orient="records")}


# ──────────────────────────── Points & Polygons ─────────────────────────────


@mcp.tool()
def load_points(filepath: str, fformat: str | None = None) -> dict:
    """Load points from file (xyz, csv, zmap_ascii, rms_attr, parquet)."""
    filepath = os.path.expanduser(filepath)
    kwargs: dict[str, Any] = {"points_file": filepath}
    if fformat:
        kwargs["fformat"] = fformat
    pts = xtgeo.points_from_file(**kwargs)
    key = _put(pts, "pts")
    df = pts.get_dataframe()
    return {
        "key": key,
        "npoints": len(df),
        "columns": list(df.columns),
        "head": df.head(10).to_dict(orient="records"),
    }


@mcp.tool()
def save_points(key: str, filepath: str, fformat: str = "xyz") -> str:
    """Export points to file (xyz, csv, rms_attr, parquet)."""
    filepath = os.path.expanduser(filepath)
    pts = _get(key)
    pts.to_file(filepath, fformat=fformat)
    return f"Points saved to {filepath}"


@mcp.tool()
def load_polygons(filepath: str, fformat: str | None = None) -> dict:
    """Load polygons from file (xyz, csv, zmap_ascii, parquet)."""
    filepath = os.path.expanduser(filepath)
    kwargs: dict[str, Any] = {"pfile": filepath}
    if fformat:
        kwargs["fformat"] = fformat
    poly = xtgeo.polygons_from_file(**kwargs)
    key = _put(poly, "poly")
    df = poly.get_dataframe()
    return {
        "key": key,
        "nrows": len(df),
        "columns": list(df.columns),
        "head": df.head(10).to_dict(orient="records"),
    }


@mcp.tool()
def save_polygons(key: str, filepath: str, fformat: str = "xyz") -> str:
    """Export polygons to file (xyz, csv, parquet)."""
    filepath = os.path.expanduser(filepath)
    poly = _get(key)
    poly.to_file(filepath, fformat=fformat)
    return f"Polygons saved to {filepath}"


# ─────────────────────── Cross-data operations ──────────────────────────────


@mcp.tool()
def surface_from_cube_slice(cube_key: str, zslice: float) -> dict:
    """Create a new surface by slicing a cube at a constant Z value."""
    cube = _get(cube_key)
    surf = xtgeo.surface_from_cube(cube, zslice=zslice)
    key = _put(surf, "surf")
    return {
        "key": key,
        "ncol": surf.ncol,
        "nrow": surf.nrow,
        "values_min": float(np.nanmin(surf.values)),
        "values_max": float(np.nanmax(surf.values)),
        "values_mean": float(np.nanmean(surf.values)),
    }


@mcp.tool()
def grid_compute_dz(grid_key: str) -> dict:
    """Compute cell thickness (DZ) for a 3D grid and store as a grid property."""
    grid = _get(grid_key)
    dz = grid.get_dz()
    key = _put(dz, "gprop")
    return {
        "key": key,
        "name": dz.name,
        "values_min": float(np.nanmin(dz.values)),
        "values_max": float(np.nanmax(dz.values)),
        "values_mean": float(np.nanmean(dz.values)),
    }


@mcp.tool()
def grid_compute_bulk_volume(grid_key: str) -> dict:
    """Compute bulk volume for each cell in a 3D grid."""
    grid = _get(grid_key)
    bv = grid.get_bulk_volume()
    key = _put(bv, "gprop")
    return {
        "key": key,
        "name": bv.name,
        "values_min": float(np.nanmin(bv.values)),
        "values_max": float(np.nanmax(bv.values)),
        "values_mean": float(np.nanmean(bv.values)),
    }


# ──────────────────────────── Convenience ───────────────────────────────────


@mcp.tool()
def xtgeo_version() -> str:
    """Return the installed xtgeo version."""
    return xtgeo.__version__


# ──────────────────────────── Entry point ───────────────────────────────────

if __name__ == "__main__":
    mcp.run()
