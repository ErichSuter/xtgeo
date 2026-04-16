from dataclasses import FrozenInstanceError
from io import BytesIO, StringIO

import numpy as np
import pytest

import xtgeo
from xtgeo.surface._regsurf_gxf_parser import GXFSurface


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


class TestGXFParsing:
    """Tests for reading and parsing GXF content."""

    def test_valid_with_extension_keys(self, valid_gxf_content: str) -> None:
        with pytest.warns(UserWarning) as recorded:
            result = GXFSurface.from_file(gxf_stream(valid_gxf_content))

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

        expected_mask = np.array(
            [[False, False], [False, True], [False, False]]
        )
        np.testing.assert_array_equal(result.values.mask, expected_mask)

    def test_from_file_path_input(
        self, tmp_path, valid_gxf_content: str
    ) -> None:
        path = tmp_path / "surface.gxf"
        path.write_text(valid_gxf_content)

        with pytest.warns(UserWarning):
            result = GXFSurface.from_file(path)

        assert result.ncol == 3

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            GXFSurface.from_file("this_file_does_not_exist.gxf")

    @pytest.mark.parametrize("count", [5, 7])
    def test_grid_value_count_must_match_ncol_times_nrow(
        self, count: int
    ) -> None:
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

        with pytest.raises(
            ValueError, match="Number of values in #GRID section"
        ):
            GXFSurface.from_file(gxf_stream(content))

    def test_missing_mandatory_key_raises(self) -> None:
        """Missing #GRID (a truly required key) must raise."""
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

        with pytest.raises(ValueError, match="Missing mandatory key"):
            GXFSurface.from_file(gxf_stream(content))

    def test_optional_keys_default_with_warnings(self) -> None:
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
            result = GXFSurface.from_file(gxf_stream(content))

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

        # All 6 values should be present and valid, no masking due to
        # missing #DUMMY
        assert result.values.count() == 6

    def test_no_dummy_means_no_masking(self) -> None:
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
            result = GXFSurface.from_file(gxf_stream(content))

        assert result.values.count() == 4
        assert not np.any(result.values.mask)

    def test_unknown_single_hash_key_warns_and_skips(self) -> None:
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
            result = GXFSurface.from_file(gxf_stream(content))

        assert result.ncol == 3
        assert result.nrow == 2

    def test_duplicate_key_raises(self) -> None:
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
            GXFSurface.from_file(gxf_stream(content))

    def test_missing_grid_key_raises(self) -> None:
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
            GXFSurface.from_file(gxf_stream(content))

    def test_sense_key_warns_about_orientation(self) -> None:
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
            result = GXFSurface.from_file(gxf_stream(content))

        assert result.ncol == 2


class TestGXFFileRoundtrip:
    """Tests for write-then-read roundtrips via file streams."""

    def test_roundtrip_stringio(self) -> None:
        values = np.ma.array(
            [[1.0, 4.0], [2.0, 9999.0], [3.0, 6.0]],
            mask=[[False, False], [False, True], [False, False]],
        )
        gxf = GXFSurface(
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

        re_read = GXFSurface.from_file(stream)

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

    def test_roundtrip_bytesio(self) -> None:
        values = np.ma.array(
            [[1.0, 2.0], [3.0, 9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = GXFSurface(
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

        re_read = GXFSurface.from_file(stream)

        np.testing.assert_allclose(re_read.values.data, gxf.values.data)
        np.testing.assert_array_equal(re_read.values.mask, gxf.values.mask)

    def test_roundtrip_wide_grid(self) -> None:
        """Roundtrip a grid wide enough to require line wrapping."""
        ncol = 40
        nrow = 2
        values = np.ma.arange(ncol * nrow, dtype=np.float64).reshape(
            (ncol, nrow)
        )
        gxf = GXFSurface(
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

        re_read = GXFSurface.from_file(stream)
        assert re_read.ncol == ncol
        assert re_read.nrow == nrow
        np.testing.assert_allclose(re_read.values.data, gxf.values.data)


class TestGXFWriter:
    """Tests for GXF output format constraints."""

    def test_line_length_at_most_80_chars(self) -> None:
        """GXF spec requires all lines <= 80 characters."""
        ncol = 20
        nrow = 3
        values = np.ma.arange(ncol * nrow, dtype=np.float64).reshape(
            (ncol, nrow)
        )
        gxf = GXFSurface(
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
        re_read = GXFSurface.from_file(stream)
        np.testing.assert_allclose(re_read.values.data, gxf.values.data)


class TestGXFDataclass:
    """Tests for GXFData dataclass properties."""

    def test_frozen(self) -> None:
        gxf = GXFSurface(
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


class TestFileFormatVerification:
    """Tests for file format detection and validation."""

    def test_valid_gxf_file_path(self, tmp_path) -> None:
        """Format verification should succeed for a valid .gxf file."""
        content = """#POINTS
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
#GRID
1 2 3 4
"""
        path = tmp_path / "test.gxf"
        path.write_text(content)

        result = GXFSurface.from_file(path)
        assert result.ncol == 2
        assert result.nrow == 2

    def test_non_gxf_content_raises(self, tmp_path) -> None:
        """
        Format verification should fail for non-GXF content (missing #ROW).
        Format verification checks for presence of some mandatory keys.
        There is also a check for mandatory keys, but that comes afterwards
        and will not be reached if format verification fails as expected.
        """
        content = """
#POINTS
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
1 2 3 4
"""
        path = tmp_path / "test.gxf"
        path.write_text(content)

        with pytest.raises(
            ValueError,
            match="does not match format detected from file contents",
        ):
            GXFSurface.from_file(path)

    def test_non_gxf_extension_and_content_raises(self, tmp_path) -> None:
        """Non-.gxf file path with non-GXF content should fail."""
        content = """
This is not a GXF key
"""
        path = tmp_path / "test.txt"
        path.write_text(content)

        with pytest.raises(
            ValueError,
            match="does not match format detected from file contents",
        ):
            GXFSurface.from_file(path)


class TestDummyTypePreservation:
    """#DUMMY value should retain the type (int or float) from the input."""

    @staticmethod
    def _minimal_gxf(dummy_literal: str) -> str:
        return f"""
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
"{dummy_literal}"
#GRID
1 2 3 {dummy_literal}
"""

    def test_dummy_int_from_file(self) -> None:
        """An integer dummy like -9999 should be parsed as int."""
        content = self._minimal_gxf("-9999")
        result = GXFSurface.from_file(gxf_stream(content))

        assert result.dummy == -9999
        assert isinstance(result.dummy, int)

    def test_dummy_float_from_file(self) -> None:
        """A float dummy like -9999.0 should be parsed as float."""
        content = self._minimal_gxf("-9999.0")
        result = GXFSurface.from_file(gxf_stream(content))

        assert result.dummy == pytest.approx(-9999.0)
        assert isinstance(result.dummy, float)

    def test_dummy_float_scientific_from_file(self) -> None:
        """A scientific-notation dummy like 1e33 should be parsed as float."""
        content = self._minimal_gxf("1e33")
        result = GXFSurface.from_file(gxf_stream(content))

        assert result.dummy == pytest.approx(1e33)
        assert isinstance(result.dummy, float)

    def test_dummy_int_masking(self) -> None:
        """Grid values equal to an int dummy should be masked."""
        content = self._minimal_gxf("-9999")
        result = GXFSurface.from_file(gxf_stream(content))

        assert result.values.count() == 3
        assert result.values.mask.any()

    def test_dummy_float_masking(self) -> None:
        """Grid values equal to a float dummy should be masked."""
        content = self._minimal_gxf("-9999.0")
        result = GXFSurface.from_file(gxf_stream(content))

        assert result.values.count() == 3
        assert result.values.mask.any()

    def test_dummy_int_roundtrip_file(self) -> None:
        """Int dummy type should survive a write-then-read roundtrip."""
        values = np.ma.array(
            [[1.0, 3.0], [2.0, -9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = GXFSurface(
            ncol=2, nrow=2, xinc=1.0, yinc=1.0,
            xori=0.0, yori=0.0, rotation=0.0,
            dummy=-9999,
            values=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFSurface.from_file(stream)
        assert re_read.dummy == -9999
        assert isinstance(re_read.dummy, int)
        np.testing.assert_array_equal(re_read.values.mask, gxf.values.mask)

    def test_dummy_float_roundtrip_file(self) -> None:
        """Float dummy type should survive a write-then-read roundtrip."""
        values = np.ma.array(
            [[1.0, 3.0], [2.0, -9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = GXFSurface(
            ncol=2, nrow=2, xinc=1.0, yinc=1.0,
            xori=0.0, yori=0.0, rotation=0.0,
            dummy=-9999.0,
            values=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFSurface.from_file(stream)
        assert re_read.dummy == pytest.approx(-9999.0)
        assert isinstance(re_read.dummy, float)
        np.testing.assert_array_equal(re_read.values.mask, gxf.values.mask)


class TestRegularSurfaceIntegration:
    """Tests for xtgeo RegularSurface GXF integration."""

    def test_from_file(self, valid_gxf_content: str) -> None:
        with pytest.warns(UserWarning):
            surf = xtgeo.surface_from_file(
                gxf_stream(valid_gxf_content), fformat="gxf"
            )

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

    def test_to_file_roundtrip(self) -> None:
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
            re_read.values.filled(np.nan),
            surf.values.filled(np.nan),
            equal_nan=True,
        )

    def test_guess_format_by_extension(self, tmp_path) -> None:
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
