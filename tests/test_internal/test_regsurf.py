"""Test some basic _internal functions which are in C++ and use the pybind11 method.

Some basic methods are tested here, while the more complex ones are tested in an
integrated manner in the other more general tests (like testing surfaces, cubes,
3D grids)

This module focus on testing the C++ functions that are used in the "regsurf"
name space.

"""

import numpy as np
import pytest

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo.common.log import null_logger

logger = null_logger(__name__)


def test_find_cell_range_simple_norotated():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=2, ncol=3, nrow=4, rotation=0
    )

    assert isinstance(surf, xtgeo.RegularSurface)

    xmin, xmax = surf.xmin, surf.xmax
    ymin, ymax = surf.ymin, surf.ymax

    print(xmin, xmax, ymin, ymax)
    print(surf.rotation)

    result = _internal.regsurf.find_cell_range(
        xmin,
        xmax,
        ymin,
        ymax,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.rotation,
        surf.ncol,
        surf.nrow,
        0,
    )

    assert result == (0, 2, 0, 3)


def test_find_cell_range_simple_rotated1():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=3, nrow=4, rotation=45
    )

    assert isinstance(surf, xtgeo.RegularSurface)

    xmin, xmax = surf.xmin, surf.xmax
    ymin, ymax = surf.ymin, surf.ymax

    print(xmin, xmax, ymin, ymax)
    print(surf.rotation)

    result = _internal.regsurf.find_cell_range(
        xmin,
        xmax,
        ymin,
        ymax,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.rotation,
        surf.ncol,
        surf.nrow,
        0,
    )

    assert result == (0, 2, 0, 3)


def test_find_cell_range_simple_rotated2():
    surf = xtgeo.RegularSurface(
        xori=1000, yori=2000, xinc=1, yinc=1, ncol=6, nrow=5, rotation=30
    )

    xmin, xmax = 1000 - 0.5, 1000 + 0.2
    ymin, ymax = 2000 + 4.2, 2000 + 4.5

    result = _internal.regsurf.find_cell_range(
        xmin,
        xmax,
        ymin,
        ymax,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.rotation,
        surf.ncol,
        surf.nrow,
        0,
    )
    assert result == (2, 2, 4, 4)

    result = _internal.regsurf.find_cell_range(
        xmin,
        xmax,
        ymin,
        ymax,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.rotation,
        surf.ncol,
        surf.nrow,
        1,
    )
    assert result == (1, 3, 3, 4)


def test_find_cell_range_simple_outside():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=3, nrow=4, rotation=45
    )

    assert isinstance(surf, xtgeo.RegularSurface)

    xmin, xmax = 1000, 2000
    ymin, ymax = 99, 1001

    result = _internal.regsurf.find_cell_range(
        xmin,
        xmax,
        ymin,
        ymax,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.rotation,
        surf.ncol,
        surf.nrow,
        0,
    )

    assert result == (2, 2, 0, 1)


def test_get_outer_corners():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=3, nrow=4, rotation=30
    )

    result = _internal.regsurf.get_outer_corners(
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.ncol,
        surf.nrow,
        surf.rotation,
    )

    print(result)
    assert result[0].x == pytest.approx(0.0)
    assert result[0].y == pytest.approx(0.0)
    assert result[1].x == pytest.approx(2.59808, rel=0.01)
    assert result[1].y == pytest.approx(1.5)
    assert result[2].x == pytest.approx(-2.0)
    assert result[2].y == pytest.approx(3.46410, rel=0.01)
    assert result[3].x == pytest.approx(0.59808, rel=0.01)
    assert result[3].y == pytest.approx(4.96410, rel=0.01)


def test_get_xy_from_ij():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=6, nrow=5, rotation=30
    )

    point = _internal.regsurf.get_xy_from_ij(
        2,
        4,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.ncol,
        surf.nrow,
        surf.rotation,
    )

    print(point.x, point.y)
    assert point.x == pytest.approx(-0.2679491924)
    assert point.y == pytest.approx(4.4641016151)


@pytest.fixture(scope="module", name="get_drogondata")
def fixture_get_drogondata(testdata_path):
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/drogon/2/geogrid.roff")
    poro = xtgeo.gridproperty_from_file(
        f"{testdata_path}/3dgrids/drogon/2/geogrid--phit.roff"
    )
    facies = xtgeo.gridproperty_from_file(
        f"{testdata_path}/3dgrids/drogon/2/geogrid--facies.roff"
    )

    surf = xtgeo.surface_from_file(
        f"{testdata_path}/surfaces/drogon/1/01_topvolantis.gri"
    )
    return grid, poro, facies, surf


def test_sample_grid3d_layer(get_drogondata):
    grid, poro, facies, surf = get_drogondata

    logger.info("Sample the grid...")
    iindex, jindex, depth_top, depth_bot, inactive = (
        _internal.regsurf.sample_grid3d_layer(
            surf.ncol,
            surf.nrow,
            surf.xori,
            surf.yori,
            surf.xinc,
            surf.yinc,
            surf.rotation,
            grid.ncol,
            grid.nrow,
            grid.nlay,
            grid._coordsv,
            grid._zcornsv,
            grid._actnumsv,
            8,  # 8 is the depth index
            2,
            -1,  # number of threads for OpenMP; -1 means let the system decide
        )
    )
    logger.info("Sample the grid... DONE")

    mask = iindex == -1
    iindex = np.where(iindex == -1, 0, iindex)
    jindex = np.where(jindex == -1, 0, jindex)

    depthmap = surf.copy()
    depthmap.values = depth_top

    poromap = surf.copy()
    # for each map node, I want the poro value given ii and jj
    poromap.values = poro.values[iindex, jindex, 0]
    poromap.values.mask = mask

    facimap = surf.copy()
    # for each map node, I want the poro value given ii and jj
    facimap.values = facies.values[iindex, jindex, 0]
    facimap.values.mask = mask

    assert np.allclose(poromap.values.mean(), 0.1974, atol=0.01)


@pytest.fixture(scope="module")
def keep_top_store():
    """To remember the top layer for the single-threaded case."""
    return {"keep_top": None}


@pytest.mark.parametrize("num_threads", [1, 2, 4, 8, 16])
def test_sample_grid3d_layer_num_threads(
    get_drogondata, benchmark, num_threads, keep_top_store
):
    """Benchmark the sampling of the grid for different number of threads."""
    grid, _, _, surf = get_drogondata

    def sample_grid():
        return _internal.regsurf.sample_grid3d_layer(
            surf.ncol,
            surf.nrow,
            surf.xori,
            surf.yori,
            surf.xinc,
            surf.yinc,
            surf.rotation,
            grid.ncol,
            grid.nrow,
            grid.nlay,
            grid._coordsv,
            grid._zcornsv,
            grid._actnumsv,
            8,  # 8 is the depth index
            2,
            num_threads,
        )

    _, _, top, bot, inactive = benchmark(sample_grid)

    top[np.isnan(top)] = 0
    if num_threads == 1:
        keep_top_store["keep_top"] = top

    # check if the top layer is exactly the same for all threads
    assert np.array_equal(top, keep_top_store["keep_top"])