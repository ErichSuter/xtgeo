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
    with pytest.warns(UserWarning) as recorded:
        result = GXFData.from_file(gxf_stream(valid_gxf_content))

    warning_messages = [str(w.message) for w in recorded]
    assert any("##XMAX" in message for message in warning_messages)
    assert any("##YMAX" in message for message in warning_messages)

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
    """Missing #ROWS (a truly required key) must raise."""
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


def test_optional_keys_default_with_warnings() -> None:
    """When optional keys are missing, defaults are applied with warnings."""
    content = """
#POINTS
"3"
#ROWS
"2"
#GRID
1 2 3 4 5 6
"""

    with pytest.warns(UserWarning) as recorded:
        result = GXFData.from_file(gxf_stream(content))

    msgs = [str(w.message) for w in recorded]
    assert any("#PTSEPARATION" in m and "default" in m for m in msgs)
    assert any("#RWSEPARATION" in m and "default" in m for m in msgs)
    assert any("#XORIGIN" in m and "default" in m for m in msgs)
    assert any("#YORIGIN" in m and "default" in m for m in msgs)
    assert any("#ROTATION" in m and "default" in m for m in msgs)
    assert any("#DUMMY" in m for m in msgs)

    assert result.ncol == 3
    assert result.nrow == 2
    assert result.xinc == pytest.approx(1.0)
    assert result.yinc == pytest.approx(1.0)
    assert result.xori == pytest.approx(0.0)
    assert result.yori == pytest.approx(0.0)
    assert result.rotation == pytest.approx(0.0)

    # All 6 values should be present and valid, no masking due to missing #DUMMY
    assert result.values.count() == 6


def test_no_dummy_means_no_masking() -> None:
    """Without #DUMMY, no values should be masked."""
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
"0"
#YORIGIN
"0"
#ROTATION
"0"
#GRID
1 2 3 4
"""

    with pytest.warns(UserWarning, match="#DUMMY"):
        result = GXFData.from_file(gxf_stream(content))

    assert result.values.count() == 4
    assert not np.any(result.values.mask)


def test_unknown_single_hash_key_warns_and_skips() -> None:
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
#UNKNOWN_KEY
"17"
#GRID
1 2 3 4 5 6
"""

    with pytest.warns(UserWarning, match="#UNKNOWN_KEY"):
        result = GXFData.from_file(gxf_stream(content))

    assert result.ncol == 3
    assert result.nrow == 2


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

    re_read = GXFData.from_file(stream)

    assert re_read.ncol == gxf.ncol
    assert re_read.nrow == gxf.nrow
    assert re_read.xinc == pytest.approx(gxf.xinc)
    assert re_read.yinc == pytest.approx(gxf.yinc)
    assert re_read.xori == pytest.approx(gxf.xori)
    assert re_read.yori == pytest.approx(gxf.yori)
    assert re_read.rotation == pytest.approx(gxf.rotation)
    assert re_read.dummy == pytest.approx(gxf.dummy)
    np.testing.assert_allclose(re_read.values.data, gxf.values.data)
    np.testing.assert_array_equal(re_read.values.mask, gxf.values.mask)


def test_to_file_and_from_file_roundtrip_bytesio() -> None:
    values = np.ma.array(
        [[1.0, 2.0], [3.0, 9999.0]], mask=[[False, False], [False, True]]
    )
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

    re_read = GXFData.from_file(stream)

    np.testing.assert_allclose(re_read.values.data, gxf.values.data)
    np.testing.assert_array_equal(re_read.values.mask, gxf.values.mask)


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
    re_read = GXFData.from_dict(as_dict)

    assert re_read.ncol == gxf.ncol
    assert re_read.nrow == gxf.nrow
    assert re_read.rotation == pytest.approx(gxf.rotation)
    np.testing.assert_allclose(re_read.values.data, gxf.values.data)
    np.testing.assert_array_equal(re_read.values.mask, gxf.values.mask)


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


def test_writer_line_length_at_most_80_chars() -> None:
    """GXF spec requires all lines <= 80 characters."""
    ncol = 20
    nrow = 3
    values = np.ma.arange(ncol * nrow, dtype=np.float64).reshape((ncol, nrow))
    gxf = GXFData(
        ncol=ncol,
        nrow=nrow,
        xinc=1.0,
        yinc=1.0,
        xori=0.0,
        yori=0.0,
        rotation=0.0,
        dummy=-999.0,
        values=values,
    )

    stream = StringIO()
    gxf.to_file(stream)
    stream.seek(0)
    for line in stream:
        assert len(line.rstrip("\n")) <= 80

    stream.seek(0)
    re_read = GXFData.from_file(stream)
    np.testing.assert_allclose(re_read.values.data, gxf.values.data)


def test_writer_roundtrip_wide_grid() -> None:
    """Roundtrip a grid wide enough to require line wrapping."""
    ncol = 40
    nrow = 2
    values = np.ma.arange(ncol * nrow, dtype=np.float64).reshape((ncol, nrow))
    gxf = GXFData(
        ncol=ncol,
        nrow=nrow,
        xinc=1.0,
        yinc=1.0,
        xori=0.0,
        yori=0.0,
        rotation=0.0,
        dummy=-999.0,
        values=values,
    )

    stream = StringIO()
    gxf.to_file(stream)
    stream.seek(0)

    re_read = GXFData.from_file(stream)
    assert re_read.ncol == ncol
    assert re_read.nrow == nrow
    np.testing.assert_allclose(re_read.values.data, gxf.values.data)


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
    np.testing.assert_allclose(
        surf.values.filled(np.nan), expected_data, equal_nan=True
    )


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

    re_read = xtgeo.surface_from_file(stream, fformat="gxf")

    assert re_read.ncol == surf.ncol
    assert re_read.nrow == surf.nrow
    assert re_read.xinc == pytest.approx(surf.xinc)
    assert re_read.yinc == pytest.approx(surf.yinc)
    assert re_read.xori == pytest.approx(surf.xori)
    assert re_read.yori == pytest.approx(surf.yori)
    assert re_read.rotation == pytest.approx(surf.rotation)

    np.testing.assert_allclose(
        re_read.values.filled(np.nan), surf.values.filled(np.nan), equal_nan=True
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


def test_sense_key_warns_about_orientation() -> None:
    """#SENSE key should produce a specific orientation warning."""
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
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#SENSE
"1"
#GRID
1 2 3 4
"""
    with pytest.warns(UserWarning, match="SENSE.*orientation"):
        result = GXFData.from_file(gxf_stream(content))

    assert result.ncol == 2
