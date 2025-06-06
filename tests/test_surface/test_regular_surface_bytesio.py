"""Testing BytesIO stuff for RegularSurfaces."""

import base64
import io
import pathlib
import threading

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

TESTSET1 = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
TESTSET2 = pathlib.Path("surfaces/reek/1/topreek_rota.fgr")


@pytest.mark.filterwarnings("ignore:Default values*")
def test_irapbin_bytesio_threading(default_surface):
    """Test threading for segfaults."""

    def test_xtgeo():
        stream = io.BytesIO()
        surface = xtgeo.RegularSurface(**default_surface)
        surface.to_file(stream)
        print("XTGeo succeeded")

    threading.Timer(1.0, test_xtgeo).start()


@pytest.mark.filterwarnings("ignore:Default values*")
def test_irapasc_bytesio_threading(default_surface):
    """Test threading for segfaults, Irap ASCII."""

    def test_xtgeo():
        stream = io.BytesIO()
        surface = xtgeo.RegularSurface(**default_surface)
        surface.to_file(stream, fformat="irap_ascii")
        print("XTGeo succeeded")

    threading.Timer(1.0, test_xtgeo).start()


def test_irapbin_import_bytesio(testdata_path):
    """Import Irap binary via bytesIO."""
    logger.info("Import file as BytesIO")

    with open(testdata_path / TESTSET1, "rb") as fin:
        stream = io.BytesIO(fin.read())
    print(dir(stream))
    print(type(stream.getvalue()))

    xsurf = xtgeo.surface_from_file(stream, fformat="irap_binary")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    assert abs(xsurf.values.mean() - 1698.648) < 0.01
    xsurf.describe()


def test_irapbin_export_bytesio(tmp_path, testdata_path):
    """Export Irap binary to bytesIO, then read again."""
    logger.info("Import and export to bytesio")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET1, fformat="irap_binary")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    assert abs(xsurf.values.mean() - 1698.648) < 0.01
    xsurf.describe()
    xsurf.to_file(tmp_path / "bytesio1.gri", fformat="irap_binary")

    xsurf.values -= 200

    stream = io.BytesIO()

    xsurf.to_file(stream, fformat="irap_binary")

    xsurfx = xtgeo.surface_from_file(stream, fformat="irap_binary")
    logger.info("XSURFX mean %s", xsurfx.values.mean())

    with open(tmp_path / "bytesio2.gri", "wb") as myfile:
        myfile.write(stream.getvalue())

    xsurf1 = xtgeo.surface_from_file(tmp_path / "bytesio1.gri", fformat="irap_binary")
    xsurf2 = xtgeo.surface_from_file(tmp_path / "bytesio2.gri", fformat="irap_binary")
    assert abs(xsurf1.values.mean() - xsurf2.values.mean() - 200) < 0.001

    stream.close()


def test_irapascii_export_import_bytesio(tmp_path, testdata_path):
    """Export Irap ascii to bytesIO, then read again."""
    logger.info("Import and export to bytesio")

    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET2, fformat="irap_ascii")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    assert abs(xsurf.values.mean() - 1698.648) < 0.01
    xsurf.to_file(tmp_path / "bytesio1.fgr", fformat="irap_ascii")

    xsurf.values -= 200

    stream = io.BytesIO()

    xsurf.to_file(stream, fformat="irap_ascii")

    xsurfx = xtgeo.surface_from_file(stream, fformat="irap_ascii")
    assert xsurf.values.mean() == xsurfx.values.mean()

    with open(tmp_path / "bytesio2.fgr", "wb") as myfile:
        myfile.write(stream.getvalue())

    xsurf1 = xtgeo.surface_from_file(tmp_path / "bytesio1.fgr", fformat="irap_ascii")
    xsurf2 = xtgeo.surface_from_file(tmp_path / "bytesio2.fgr", fformat="irap_ascii")
    assert abs(xsurf1.values.mean() - xsurf2.values.mean() - 200) < 0.001

    stream.close()


def test_get_regsurfi(testdata_path):
    """Get regular surface from stream."""
    sfile = testdata_path / TESTSET1
    with open(sfile, "rb") as fin:
        stream = io.BytesIO(fin.read())

    logger.info("File is %s", sfile)
    for _itmp in range(20):
        rf = xtgeo.surface_from_file(stream, fformat="irap_binary")
        assert abs(rf.values.mean() - 1698.648) < 0.01
        print(_itmp)


def test_get_regsurff(testdata_path):
    """Get regular surface from file."""
    sfile = testdata_path / TESTSET1
    logger.info("File is %s", sfile)
    for _itmp in range(20):
        rf = xtgeo.surface_from_file(sfile, fformat="irap_binary")
        assert abs(rf.values.mean() - 1698.648) < 0.01
        print(_itmp)


def test_irapbin_load_meta_first_bytesio(testdata_path):
    """Import Irap binary via bytesIO, by just loading metadata first."""
    logger.info("Import and export...")

    with open(testdata_path / TESTSET1, "rb") as fin:
        stream = io.BytesIO(fin.read())

    xsurf = xtgeo.surface_from_file(stream, fformat="irap_binary", values=False)
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    xsurf.describe()

    xsurf.load_values()
    xsurf.describe()
    stream.close()

    # stream is now closed
    with pytest.raises(ValueError) as verr:
        xsurf = xtgeo.surface_from_file(stream, fformat="irap_binary", values=False)
    assert "I/O operation on closed file" in str(verr.value)


def test_bytesio_string_encoded(testdata_path):
    """Test a case where the string is encoded, then decoded."""
    with open(testdata_path / TESTSET1, "rb") as fin:
        stream = io.BytesIO(fin.read())

    mystream = stream.read()

    # this mimics data from a browser that are base64 encoded
    encodedstream = base64.urlsafe_b64encode(mystream).decode("utf-8")
    assert isinstance(encodedstream, str)

    # now decode this and read
    decodedstream = base64.urlsafe_b64decode(encodedstream)
    assert isinstance(decodedstream, bytes)

    content_string = io.BytesIO(decodedstream)
    xsurf = xtgeo.surface_from_file(content_string, fformat="irap_binary")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451


def test_export_import_hdf5_bytesio(tmp_path, testdata_path):
    """Test hdf5 format via memory streams."""
    # just the input, and save as hdf5
    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET2, fformat="irap_ascii")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    xsurf.to_hdf(tmp_path / "surf.hdf")

    xsurf2 = xtgeo.surface_from_file(tmp_path / "surf.hdf", fformat="hdf")
    assert xsurf2.ncol == 554

    stream = io.BytesIO()
    xsurf.to_hdf(stream)

    xsurf3 = xtgeo.surface_from_file(stream, fformat="hdf")
    assert xsurf3.ncol == 554

    assert xsurf3.values.mean() == pytest.approx(xsurf.values.mean())

    xsurf4 = xtgeo.surface_from_file(stream, fformat="hdf")
    assert xsurf4.values.mean() == pytest.approx(xsurf.values.mean())


def test_export_import_ijxyz_bytesio(tmp_path, testdata_path):
    """Test ijxyz format via files and memory streams."""
    # just the input, and save as hdf5
    xsurf = xtgeo.surface_from_file(testdata_path / TESTSET2, fformat="irap_ascii")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451

    ijxyzfile = tmp_path / "ijxyz_tset2.fgr"
    xsurf.to_file(ijxyzfile, fformat="ijxyz")

    xsurf2 = xtgeo.surface_from_file(ijxyzfile, fformat="ijxyz")

    # if surface is undefined in areas at border, the ijxyz format will not be
    # able to know the original ncol, nrow, so a reduction may be expected
    assert xsurf2.ncol == 525
    assert xsurf2.nrow == 442
    assert xsurf2.values.mean() == pytest.approx(xsurf.values.mean())

    stream = io.BytesIO()

    xsurf.to_file(stream, fformat="ijxyz")

    xsurf3 = xtgeo.surface_from_file(stream, fformat="ijxyz")
    assert xsurf3.ncol == xsurf2.ncol
    assert xsurf3.values.mean() == pytest.approx(xsurf2.values.mean())
