import os

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

# Open the GXF file
print(os.getcwd())
gdal.UseExceptions()
#gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_minimum.gxf"         # OK
# gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_minimum_misc_testing.gxf"         # OK
# gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_test.gxf"            # NOK
# gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_test_no_header.gxf"  # NOK
# gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_test_no_apostrophes.gxf" # NOK
# gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_test_no_header_no_apostrophes.gxf" # OK

# gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_test_2_rotated.gxf"         # NOK
gxf_file = "../xtgeo-testdata/surfaces/etc/fdata_test_2_rotated_no_header.gxf"         # OK?

dataset = gdal.Open(gxf_file)

if dataset is not None:
    # Get dimensions
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # Get geotransform (affine transformation coefficients)
    geotransform = dataset.GetGeoTransform()

    # Get projection
    projection = dataset.GetProjection()

    # Read the data into a numpy array
    band = dataset.GetRasterBand(1)  # GXF typically has one band
    data = band.ReadAsArray()

    # Get no-data value if it exists
    nodata = band.GetNoDataValue()
    if nodata is not None:
        # Replace no-data values with NaN for visualization
        data = np.where(data == nodata, np.nan, data)


    metadata = dataset.GetMetadata()  # Get metadata to check for header info

    raw_info = dataset.GetMetadataItem("RAW_INFO")
    if raw_info:
        print(f"RAW_INFO metadata: {raw_info}")
    else:
        print("No RAW_INFO metadata found.")

    # raw_info = dataset.GetRawInfo()
    # if raw_info:
    #     print(f"Raw info: {raw_info}")
    # else:
    #     print("No raw info found.")

    position = dataset.GetMetadataItem("POSITION")
    if position:
        print(f"Position metadata: {position}")
    else:
        print("No POSITION metadata found.")

    # position = dataset.GetRawPosition()
    # if position:
    #     print(f"Position from GetRawPosition(): {position}")
    # else:
    #     print("GetRawPosition() did not return a valid position.")


    # Now you can work with the data array
    print(f"Opened GXF file: {gxf_file}")
    print(f"Dimensions: width={width}, height={height}")
    print(f"Geotransform: {geotransform}")
    print(f"Projection: {projection}")
    print(f"Metadata: {metadata}")
    print(f"Data type: {data.dtype}")
    print(f"Data shape: {data.shape}")
    print(f"Data min: {np.nanmin(data)}, max: {np.nanmax(data)}")

    # Simple visualization
    plt.imshow(data, cmap="viridis")
    plt.colorbar()
    plt.title(f"GXF data from {gxf_file}")
    plt.show()


    # TODO:
    # GetNextFeature()
    # FlushCache(self, *args): write to disk


    # Clean up
    dataset = None
else:
    print(f"Failed to open {gxf_file}")
