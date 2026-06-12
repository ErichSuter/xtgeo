"""Unit tests for GridReader/GridWriter logic without a running ResInsight.

These inject a pre-built fake RipsApiUtils into the cached slot of the
reader/writer (``_ripsapi_utils``) so the ``load`` and ``save`` code paths run
without a live ResInsight install. Integration tests using a real instance
live in ``test_resinsight_grid.py`` gated on ``requires_resinsight``.

The fake case/project/utils objects are defined once in ``conftest.py`` and
provided through the ``ri_fakes`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from xtgeo.interfaces.resinsight._grid import (
    GridDataResInsight,
    GridReader,
    GridWriter,
)


def _reader_with_project(ri_fakes, project) -> GridReader:
    reader = GridReader(instance_or_port=None)
    reader._ripsapi_utils = ri_fakes.RipsApiUtils(project=project)  # type: ignore[assignment]
    return reader


def _writer_with_project(ri_fakes, project) -> GridWriter:
    writer = GridWriter(instance_or_port=None)
    writer._ripsapi_utils = ri_fakes.RipsApiUtils(project=project)  # type: ignore[assignment]
    return writer


def _sample_data(ri_fakes) -> GridDataResInsight:
    zcorn, coord, actnum = ri_fakes.grid_arrays()
    return GridDataResInsight(
        name="DATA",
        nx=ri_fakes.NX,
        ny=ri_fakes.NY,
        nz=ri_fakes.NZ,
        coordsv=coord,
        zcornsv=zcorn,
        actnumsv=actnum,
        filesrc="data.roff",
    )


# --- GridReader.load ----------------------------------------------------------


def test_reader_load_builds_grid_data(ri_fakes):
    case = ri_fakes.Case("EXAMPLE", file_path="/path/emerald.roff")
    reader = _reader_with_project(ri_fakes, ri_fakes.Project(cases=[case]))

    data = reader.load("EXAMPLE")

    assert data.name == "EXAMPLE"
    assert (data.nx, data.ny, data.nz) == (ri_fakes.NX, ri_fakes.NY, ri_fakes.NZ)
    assert data.filesrc == "/path/emerald.roff"
    assert data.coordsv.dtype == np.float64
    assert data.zcornsv.dtype == np.float32
    assert data.actnumsv.dtype == np.int32


def test_reader_load_empty_file_path_defaults_to_blank(ri_fakes):
    case = ri_fakes.Case("EXAMPLE", file_path="")
    reader = _reader_with_project(ri_fakes, ri_fakes.Project(cases=[case]))

    data = reader.load("EXAMPLE")

    assert data.filesrc == ""


def test_reader_load_raises_when_no_case(ri_fakes):
    reader = _reader_with_project(ri_fakes, ri_fakes.Project(cases=[]))

    with pytest.raises(RuntimeError, match="Cannot find any case with name 'MISSING'"):
        reader.load("MISSING")


# --- GridWriter.save ----------------------------------------------------------


def test_writer_save_replaces_existing_corner_point_case(ri_fakes):
    case = ri_fakes.CornerPointCase("GRID")
    project = ri_fakes.Project(cases=[case])
    writer = _writer_with_project(ri_fakes, project)

    writer.save(_sample_data(ri_fakes), gname="GRID")

    assert case.replaced_with == (ri_fakes.NX, ri_fakes.NY, ri_fakes.NZ)
    assert case.file_path == "data.roff"
    assert case.updated is True
    # Replace branch returns early; no new case created
    assert project.created_kwargs is None


def test_writer_save_creates_when_no_existing_case(ri_fakes):
    new_case = ri_fakes.NewCase()
    project = ri_fakes.Project(cases=[], create_result=new_case)
    writer = _writer_with_project(ri_fakes, project)

    writer.save(_sample_data(ri_fakes), gname="NEW")

    assert project.created_kwargs is not None
    assert project.created_kwargs["name"] == "NEW"
    assert project.created_kwargs["nx"] == ri_fakes.NX
    assert new_case.file_path == "data.roff"
    assert new_case.updated is True


def test_writer_save_creates_when_existing_case_wrong_type(ri_fakes):
    existing = ri_fakes.GridFileCase("GRID")
    new_case = ri_fakes.NewCase()
    project = ri_fakes.Project(cases=[existing], create_result=new_case)
    writer = _writer_with_project(ri_fakes, project)

    writer.save(_sample_data(ri_fakes), gname="GRID")

    # Wrong-type existing case is not replaced; a new case is created instead
    assert project.created_kwargs is not None
    assert new_case.updated is True


def test_writer_save_wraps_exceptions(ri_fakes):
    project = ri_fakes.Project(cases=[], create_raises=ValueError("boom"))
    writer = _writer_with_project(ri_fakes, project)

    with pytest.raises(RuntimeError, match="Failed to save ResInsight case data: boom"):
        writer.save(_sample_data(ri_fakes), gname="NEW")
