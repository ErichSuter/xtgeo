# ResInsight Interface Overview

## Purpose

`src/xtgeo/interfaces/resinsight` is XTGeo's bridge to ResInsight through the optional `rips` Python package. The implemented domain surface is corner-point grid exchange: reading a ResInsight case into an `xtgeo.Grid`, and writing an `xtgeo.Grid` back to ResInsight as a corner-point case.

The user-facing entry points are mostly exposed from `xtgeo.grid3d.grid`:

- `xtgeo.grid_from_resinsight(instance_or_port, case_name, find_last=True)`
- `Grid.to_resinsight(instance_or_port, gname, find_last=True)`

Both wrappers delegate into `xtgeo.interfaces.resinsight`.

## Package Layout

`__init__.py` exports the public API for this interface package:

- `GridDataResInsight`
- `GridReader`
- `GridWriter`
- `RipsApiUtils`
- `rips`

`_rips_package.py` handles optional dependency loading. It tries to import `rips`, resolves the required symbols (`Case`, `Instance`, `Project`), exposes type aliases, and enforces `MIN_RIPS_VERSION = "2026.2"` through `require_rips()`.

`rips_utils.py` contains connection and project utilities. `RipsApiUtils` can connect using:

- `None`, to auto-discover a running ResInsight instance
- an `int`, to connect to a specific gRPC port
- an existing `rips.Instance`, to reuse it directly

It exposes `instance`, `project`, `save_project()`, `close_project()`, `terminate()`, `launch_instance()`, and `find_instance()`.

`_resinsight_base.py` contains `_BaseResInsightDataRW`, a shared base for readers and writers. It stores `instance_or_port`, lazily creates and caches `RipsApiUtils`, and provides `get_ripsapi_utils()`, `get_instance()`, `get_project()`, and `get_case(case_name, find_last=True)`.

`get_case()` scans `project.cases()` by name. ResInsight case names are not unique, so `find_last=True` selects the last matching case and `find_last=False` selects the first.

`_grid.py` contains the grid exchange implementation.

## Grid Data Container

`GridDataResInsight` is a frozen dataclass used as an intermediate container between ResInsight/rips and XTGeo. It stores:

- `name`
- `nx`, `ny`, `nz`
- `coordsv`, the flat corner-point COORD array as `float64`
- `zcornsv`, the flat ZCORN array as `float32`
- `actnumsv`, the flat ACTNUM array as `int32`
- `filesrc`

Important methods:

- `__post_init__()` validates that the flat array lengths match the declared grid dimensions.
- `__eq__()` compares scalar fields normally and array fields with `np.array_equal()`.
- `to_xtgeo_grid()` converts the stored arrays into an `xtgeo.Grid` using `EGrid.default_settings_grid()` and `grid_from_ecl_grid()`.
- `from_xtgeo_grid()` converts an existing `xtgeo.Grid` into `GridDataResInsight` through `EGrid.from_xtgeo_grid()`.

Note: `GridDataResInsight` is `frozen=True`, but its NumPy array fields are not made read-only. Attribute rebinding is blocked, but array contents are still mutable through the arrays themselves.

## Reading From ResInsight

`GridReader(instance_or_port).load(case_name, find_last=True)` reads a grid from ResInsight.

The flow is:

1. Resolve a case by name using `_BaseResInsightDataRW.get_case()`.
2. Raise `RuntimeError` if no matching case exists.
3. Call `case.export_corner_point_grid()`.
4. Convert the returned arrays to expected dtypes.
5. Return a `GridDataResInsight` instance.

`xtgeo.grid_from_resinsight(...)` then calls `GridDataResInsight.to_xtgeo_grid()` to return a populated `xtgeo.Grid`.

## Writing To ResInsight

`GridWriter(instance_or_port).save(data, gname, find_last=True)` writes `GridDataResInsight` to ResInsight.

The flow is:

1. Resolve an existing case named `gname`.
2. If the existing case is a `CornerPointCase`, call `replace_corner_point_grid(...)`, set `file_path`, call `update()`, and return.
3. If no case exists, create a new corner-point case with `project.create_corner_point_grid(...)`.
4. If a case exists but is not compatible with grid data, log a warning and create a new corner-point case with the same name.
5. Wrap unexpected failures in `RuntimeError("Failed to save ResInsight case data: ...")`.

`Grid.to_resinsight(...)` first builds `GridDataResInsight.from_xtgeo_grid(...)`, then delegates to `GridWriter.save(...)`.

## Tests

The tests in `tests/test_interfaces/test_resinsight` are split into unit tests and integration-style tests:

- Unit tests use fake rips/project/case objects from `conftest.py` and do not require a running ResInsight instance.
- Integration tests are marked `requires_resinsight` and exercise live ResInsight behavior.
- Public API tests cover `xtgeo.grid_from_resinsight(...)` and `Grid.to_resinsight(...)`, including duplicate case name handling through `find_last` and grid roundtrips.
