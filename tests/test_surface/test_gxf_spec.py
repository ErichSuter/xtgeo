import pathlib
from io import StringIO

import pytest

import xtgeo

# the minimal should plot with 0 at lower left and 35 at upper right
MINIMAL_GXF = """\
#POINTS
6
#ROWS
4
#ROTATION
0.0
#DUMMY
2
#GRID
0 1 2 3 4 5
10 11 12 13 14 15
20 21 22 23 24 25
30 31 32 33 34 35
"""


def test_read_minimal_gxf(tmp_path):
    """Test reading minimal GXF data from a string"""

    # dump the minimal GXF data to a file and read it
    gxf_data = pathlib.Path(tmp_path) / "minimal.gxf"
    with open(gxf_data, "w") as f:
        f.write(MINIMAL_GXF)

    gxf = xtgeo.surface_from_file(gxf_data, fformat="gxf")
    # assert isinstance(gxf_data, xtgeo.surface.RegularSurface)
    # assert gxf_data.ncol == 6
    # assert gxf_data.nrow == 4
    # assert gxf_data.values[0, 0] == 0.0
    # assert gxf_data.values[-1, -1] == 35.0
    gxf.quickplot()


def test_read_gxf_data(testdata_path):
    """Test reading GXF data from a file"""
    tpath = pathlib.Path(testdata_path)
    gxf_file = tpath / "surfaces" / "etc" / "fdata_test2.gxf"
    gxf_data = xtgeo.surface_from_file(gxf_file, fformat="gxf")
    print(gxf_data)
    gxf_data.quickplot()
