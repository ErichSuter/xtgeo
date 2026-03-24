import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

# Open the GXF file
gxf_file = "path/to/your_file.gxf"
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

    # Now you can work with the data array
    print(f"Data shape: {data.shape}")
    print(f"Data min: {np.nanmin(data)}, max: {np.nanmax(data)}")

    # Simple visualization
    plt.imshow(data, cmap="viridis")
    plt.colorbar()
    plt.title(f"GXF data from {gxf_file}")
    plt.show()

    # Clean up
    dataset = None
else:
    print(f"Failed to open {gxf_file}")
