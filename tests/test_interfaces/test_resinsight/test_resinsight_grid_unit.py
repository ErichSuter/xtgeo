"""Unit tests for GridReader/GridWriter logic without a running ResInsight.

These inject a pre-built fake RipsApiUtils into the cached slot of the
reader/writer (``_ripsapi_utils``) so the ``load`` and ``save`` code paths run
without a live ResInsight install. Integration tests using a real instance
live in ``test_resinsight_grid.py`` gated on ``requires_resinsight``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xtgeo.interfaces.resinsight._grid import (
    GridDataResInsight,
    GridReader,
    GridWriter,
)

NX = NY = NZ = 2
COORD_SIZE = (NX + 1) * (NY + 1) * 6
ZCORN_SIZE = NX * NY * NZ * 8
ACTNUM_SIZE = NX * NY * NZ


def _grid_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (zcorn, coord, actnum) flat arrays valid for a 2x2x2 grid."""
    zcorn = np.zeros(ZCORN_SIZE, dtype=np.float32)
    coord = np.zeros(COORD_SIZE, dtype=np.float64)
    actnum = np.ones(ACTNUM_SIZE, dtype=np.int32)
    return zcorn, coord, actnum


# --- Fake ResInsight case / project / utils ----------------------------------


class _ReadableCase:
    """Fake case exposing the read API used by GridReader.load."""

    def __init__(self, name: str, file_path: str = "src.roff") -> None:
        self.name = name
        self.file_path = file_path

    def export_corner_point_grid(self):
        zcorn, coord, actnum = _grid_arrays()
        return zcorn, coord, actnum, NX, NY, NZ


class CornerPointCase:
    """Fake replaceable case; class name matters (GridWriter checks it)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.file_path = ""
        self.replaced_with: tuple | None = None
        self.updated = False

    def replace_corner_point_grid(self, nx, ny, nz, coord, zcorn, actnum):
        self.replaced_with = (nx, ny, nz)

    def update(self) -> None:
        self.updated = True


class GridFileCase:
    """Fake case of a non-replaceable type (different class name)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _NewCase:
    def __init__(self) -> None:
        self.file_path = ""
        self.updated = False

    def update(self) -> None:
        self.updated = True


class _FakeProject:
    def __init__(
        self,
        cases=None,
        create_result=None,
        create_raises: Exception | None = None,
    ) -> None:
        self._cases = cases or []
        self._create_result = create_result
        self._create_raises = create_raises
        self.created_kwargs: dict | None = None

    def cases(self):
        return self._cases

    def create_corner_point_grid(self, **kwargs):
        if self._create_raises is not None:
            raise self._create_raises
        self.created_kwargs = kwargs
        return self._create_result


class _FakeUtils:
    def __init__(self, project: _FakeProject) -> None:
        self.instance = SimpleNamespace(project=project)
        self._project = project

    @property
    def project(self) -> _FakeProject:
        return self._project


def _reader_with_project(project: _FakeProject) -> GridReader:
    reader = GridReader(instance_or_port=None)
    reader._ripsapi_utils = _FakeUtils(project)  # type: ignore[assignment]
    return reader


def _writer_with_project(project: _FakeProject) -> GridWriter:
    writer = GridWriter(instance_or_port=None)
    writer._ripsapi_utils = _FakeUtils(project)  # type: ignore[assignment]
    return writer


def _sample_data() -> GridDataResInsight:
    zcorn, coord, actnum = _grid_arrays()
    return GridDataResInsight(
        name="DATA",
        nx=NX,
        ny=NY,
        nz=NZ,
        coordsv=coord,
        zcornsv=zcorn,
        actnumsv=actnum,
        filesrc="data.roff",
    )


# --- GridReader.load ----------------------------------------------------------


def test_reader_load_builds_grid_data():
    case = _ReadableCase("EXAMPLE", file_path="/path/emerald.roff")
    reader = _reader_with_project(_FakeProject(cases=[case]))

    data = reader.load("EXAMPLE")

    assert data.name == "EXAMPLE"
    assert (data.nx, data.ny, data.nz) == (NX, NY, NZ)
    assert data.filesrc == "/path/emerald.roff"
    assert data.coordsv.dtype == np.float64
    assert data.zcornsv.dtype == np.float32
    assert data.actnumsv.dtype == np.int32


def test_reader_load_empty_file_path_defaults_to_blank():
    case = _ReadableCase("EXAMPLE", file_path="")
    reader = _reader_with_project(_FakeProject(cases=[case]))

    data = reader.load("EXAMPLE")

    assert data.filesrc == ""


def test_reader_load_raises_when_no_case():
    reader = _reader_with_project(_FakeProject(cases=[]))

    with pytest.raises(RuntimeError, match="Cannot find any case with name 'MISSING'"):
        reader.load("MISSING")


# --- GridWriter.save ----------------------------------------------------------


def test_writer_save_replaces_existing_corner_point_case():
    case = CornerPointCase("GRID")
    project = _FakeProject(cases=[case])
    writer = _writer_with_project(project)

    writer.save(_sample_data(), gname="GRID")

    assert case.replaced_with == (NX, NY, NZ)
    assert case.file_path == "data.roff"
    assert case.updated is True
    # Replace branch returns early; no new case created
    assert project.created_kwargs is None


def test_writer_save_creates_when_no_existing_case():
    new_case = _NewCase()
    project = _FakeProject(cases=[], create_result=new_case)
    writer = _writer_with_project(project)

    writer.save(_sample_data(), gname="NEW")

    assert project.created_kwargs is not None
    assert project.created_kwargs["name"] == "NEW"
    assert project.created_kwargs["nx"] == NX
    assert new_case.file_path == "data.roff"
    assert new_case.updated is True


def test_writer_save_creates_when_existing_case_wrong_type():
    existing = GridFileCase("GRID")
    new_case = _NewCase()
    project = _FakeProject(cases=[existing], create_result=new_case)
    writer = _writer_with_project(project)

    writer.save(_sample_data(), gname="GRID")

    # Wrong-type existing case is not replaced; a new case is created instead
    assert project.created_kwargs is not None
    assert new_case.updated is True


def test_writer_save_wraps_exceptions():
    project = _FakeProject(cases=[], create_raises=ValueError("boom"))
    writer = _writer_with_project(project)

    with pytest.raises(RuntimeError, match="Failed to save ResInsight case data: boom"):
        writer.save(_sample_data(), gname="NEW")
