from dataclasses import FrozenInstanceError
from io import BytesIO, StringIO

import numpy as np
import pytest

import xtgeo
from xtgeo.surface._regsurf_gxf_parser import GXFData


def gxf_stream(content: str) -> StringIO:
    return StringIO(content)


@pytest.fixture
def valid_gxf_content() -> str:
    return """! This is a comment line

Some free text to ignore

#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"30.0"
#RWSEPARATION
"40.0"
#XORIGIN
"1000.0"
#YORIGIN
"2000.0"
#ROTATION
"12.5"
#DUMMY
"9999999.0"
##XMAX
"9999.0"
##YMAX
"8888.0"
#GRID
1.0 2.0 3.0
4.0 9999999.0 6.0
"""


def test_from_file_valid_with_extension_keys(valid_gxf_content: str) -> None:
    with pytest.warns(UserWarning, match="##XMAX"):
        result = GXFData.from_file(gxf_stream(valid_gxf_content))

    assert result.ncol == 3
    assert result.nrow == 2
    assert result.xinc == pytest.approx(30.0)
    assert result.yinc == pytest.approx(40.0)
    assert result.xori == pytest.approx(1000.0)
    assert result.yori == pytest.approx(2000.0)
    assert result.rotation == pytest.approx(12.5)
    assert result.dummy == pytest.approx(9999999.0)

    expected_data = np.array([[1.0, 4.0], [2.0, 9999999.0], [3.0, 6.0]])
    np.testing.assert_allclose(result.values.data, expected_data)

    expected_mask = np.array([[False, False], [False, True], [False, False]])
    np.testing.assert_array_equal(result.values.mask, expected_mask)


def test_from_file_path_input(tmp_path, valid_gxf_content: str) -> None:
    path = tmp_path / "surface.gxf"
    path.write_text(valid_gxf_content)

    with pytest.warns(UserWarning):
        result = GXFData.from_file(path)

    assert result.ncol == 3


def test_from_file_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GXFData.from_file("this_file_does_not_exist.gxf")


@pytest.mark.parametrize("count", [5, 7])
def test_grid_value_count_must_match_ncol_times_nrow(count: int) -> None:
    values = " ".join(str(i) for i in range(count))
    content = f"""
#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#GRID
{values}
"""

    with pytest.raises(ValueError, match="Number of values in #GRID section"):
        GXFData.from_file(gxf_stream(content))


def test_missing_mandatory_key_raises() -> None:
    content = """
#POINTS
"3"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#GRID
1 2 3 4 5 6
"""

    with pytest.raises(ValueError, match="Missing mandatory keys"):
        GXFData.from_file(gxf_stream(content))


def test_unknown_single_hash_key_raises() -> None:
    content = """
#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#UNKNOWN
"17"
#GRID
1 2 3 4 5 6
"""

    with pytest.raises(ValueError, match="Unsupported key"):
        GXFData.from_file(gxf_stream(content))


def test_duplicate_key_raises() -> None:
    content = """
#POINTS
"3"
#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#GRID
1 2 3 4 5 6
"""

    with pytest.raises(ValueError, match="Duplicate key"):
        GXFData.from_file(gxf_stream(content))


def test_missing_grid_key_raises() -> None:
    content = """
#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
"""

    with pytest.raises(ValueError, match="Missing mandatory key '#GRID'"):
        GXFData.from_file(gxf_stream(content))


def test_to_file_and_from_file_roundtrip_stringio() -> None:
    values = np.ma.array(
        [[1.0, 4.0], [2.0, 9999.0], [3.0, 6.0]],
        mask=[[False, False], [False, True], [False, False]],
    )
    gxf = GXFData(
        ncol=3,
        nrow=2,
        xinc=10.0,
        yinc=20.0,
        xori=100.0,
        yori=200.0,
        rotation=30.0,
        dummy=9999.0,
        values=values,
    )

    stream = StringIO()
    gxf.to_file(stream)
    stream.seek(0)

    reread = GXFData.from_file(stream)

    assert reread.ncol == gxf.ncol
    assert reread.nrow == gxf.nrow
    assert reread.xinc == pytest.approx(gxf.xinc)
    assert reread.yinc == pytest.approx(gxf.yinc)
    assert reread.xori == pytest.approx(gxf.xori)
    assert reread.yori == pytest.approx(gxf.yori)
    assert reread.rotation == pytest.approx(gxf.rotation)
    assert reread.dummy == pytest.approx(gxf.dummy)
    np.testing.assert_allclose(reread.values.data, gxf.values.data)
    np.testing.assert_array_equal(reread.values.mask, gxf.values.mask)


def test_to_file_and_from_file_roundtrip_bytesio() -> None:
    values = np.ma.array([[1.0, 2.0], [3.0, 9999.0]], mask=[[False, False], [False, True]])
    gxf = GXFData(
        ncol=2,
        nrow=2,
        xinc=1.0,
        yinc=1.0,
        xori=0.0,
        yori=0.0,
        rotation=0.0,
        dummy=9999.0,
        values=values,
    )

    stream = BytesIO()
    gxf.to_file(stream)
    stream.seek(0)

    reread = GXFData.from_file(stream)

    np.testing.assert_allclose(reread.values.data, gxf.values.data)
    np.testing.assert_array_equal(reread.values.mask, gxf.values.mask)


def test_dict_roundtrip() -> None:
    values = np.ma.array(
        [[1.0, 4.0], [2.0, 9999.0], [3.0, 6.0]],
        mask=[[False, False], [False, True], [False, False]],
    )
    gxf = GXFData(
        ncol=3,
        nrow=2,
        xinc=10.0,
        yinc=20.0,
        xori=100.0,
        yori=200.0,
        rotation=30.0,
        dummy=9999.0,
        values=values,
    )

    as_dict = gxf.to_dict()
    reread = GXFData.from_dict(as_dict)

    assert reread.ncol == gxf.ncol
    assert reread.nrow == gxf.nrow
    assert reread.rotation == pytest.approx(gxf.rotation)
    np.testing.assert_allclose(reread.values.data, gxf.values.data)
    np.testing.assert_array_equal(reread.values.mask, gxf.values.mask)


def test_from_dict_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="Missing mandatory dictionary keys"):
        GXFData.from_dict(
            {
                "ncol": 2,
                "nrow": 2,
                "xinc": 1.0,
                "yinc": 1.0,
                "xori": 0.0,
                "yori": 0.0,
                "rotation": 0.0,
                "dummy": 9999.0,
            }
        )


def test_frozen_dataclass() -> None:
    gxf = GXFData(
        ncol=2,
        nrow=2,
        xinc=1.0,
        yinc=1.0,
        xori=0.0,
        yori=0.0,
        rotation=0.0,
        dummy=9999.0,
        values=np.ma.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    with pytest.raises(FrozenInstanceError):
        gxf.ncol = 3


def test_regular_surface_from_file_gxf_integration(valid_gxf_content: str) -> None:
    with pytest.warns(UserWarning):
        surf = xtgeo.surface_from_file(gxf_stream(valid_gxf_content), fformat="gxf")

    assert surf.ncol == 3
    assert surf.nrow == 2
    assert surf.xinc == pytest.approx(30.0)
    assert surf.yinc == pytest.approx(40.0)
    assert surf.xori == pytest.approx(1000.0)
    assert surf.yori == pytest.approx(2000.0)
    assert surf.rotation == pytest.approx(12.5)

    expected_data = np.array([[1.0, 4.0], [2.0, np.nan], [3.0, 6.0]])
    np.testing.assert_allclose(surf.values.filled(np.nan), expected_data, equal_nan=True)


def test_regular_surface_to_file_gxf_integration_roundtrip() -> None:
    values = np.ma.array(
        [[11.0, 44.0], [22.0, 55.0], [33.0, np.nan]],
        mask=[[False, False], [False, False], [False, True]],
    )
    surf = xtgeo.RegularSurface(
        ncol=3,
        nrow=2,
        xinc=10.0,
        yinc=20.0,
        xori=100.0,
        yori=200.0,
        rotation=15.0,
        values=values,
    )

    stream = BytesIO()
    surf.to_file(stream, fformat="gxf")
    stream.seek(0)

    reread = xtgeo.surface_from_file(stream, fformat="gxf")

    assert reread.ncol == surf.ncol
    assert reread.nrow == surf.nrow
    assert reread.xinc == pytest.approx(surf.xinc)
    assert reread.yinc == pytest.approx(surf.yinc)
    assert reread.xori == pytest.approx(surf.xori)
    assert reread.yori == pytest.approx(surf.yori)
    assert reread.rotation == pytest.approx(surf.rotation)

    np.testing.assert_allclose(
        reread.values.filled(np.nan), surf.values.filled(np.nan), equal_nan=True
    )


def test_regular_surface_guess_format_by_extension(tmp_path) -> None:
    content = """
#POINTS
"2"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"10"
#YORIGIN
"20"
#ROTATION
"0"
#DUMMY
"9999"
#GRID
1 2
3 9999
"""
    path = tmp_path / "small.gxf"
    path.write_text(content)

    surf = xtgeo.surface_from_file(path)
    assert surf.ncol == 2
    assert surf.nrow == 2
