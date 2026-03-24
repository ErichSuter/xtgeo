import os

import matplotlib.pyplot as plt
import numpy as np


def read_gxf(filename, plot=False, debug=False):
    """
    GXF reader customized for this specific file format
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r") as f:
        lines = f.readlines()

    if debug:
        print(f"Read {len(lines)} lines from file")
        # Print the first few lines to understand the format
        print("First 10 lines:")
        for i in range(min(10, len(lines))):
            print(f"{i + 1}: {lines[i].strip()}")

    # Try to identify header section
    header = {}
    data = []
    in_header = True
    header_pattern_count = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Check for common GXF header patterns
        if any(pattern in line for pattern in ["#POINTS", "#ROWS", "#GRID", "#DUMMY"]):
            header_pattern_count += 1
            parts = line.lstrip("#").split(maxsplit=1)
            if len(parts) > 1:
                key = parts[0]
                value = parts[1]
                header[key] = value
                if debug and header_pattern_count <= 5:
                    print(f"Found header: {key} = {value}")
            continue

        # If it looks like a row of data, process it
        try:
            values = line.split()
            row_data = [float(x) for x in values]
            data.append(row_data)
        except ValueError:
            if debug and len(header) < 10:
                print(f"Line {i + 1} not recognized as header or data: {line}")

    if debug:
        print(f"Found {len(header)} header entries")
        print(f"Parsed {len(data)} data rows")

    # Group data by row length
    data_by_length = {}
    for row in data:
        length = len(row)
        if length not in data_by_length:
            data_by_length[length] = []
        data_by_length[length].append(row)

    if debug:
        print("Data grouped by row length:")
        for length, rows in sorted(data_by_length.items()):
            print(f"  Length {length}: {len(rows)} rows")

    # Use the most common row length for the main data array
    main_length = max(data_by_length.keys(), key=lambda k: len(data_by_length[k]))
    main_data = np.array(data_by_length[main_length])

    if debug:
        print(
            f"Using data with length {main_length} ({len(data_by_length[main_length])} rows)"
        )
        if main_data.size > 0:
            print(f"Main data shape: {main_data.shape}")
            print(f"Data sample: {main_data[:3, :]}")

    # Try to determine grid dimensions
    nrows = None
    ncols = None

    if "ROWS" in header:
        try:
            nrows = int(header["ROWS"])
        except ValueError:
            pass

    if "POINTS" in header:
        try:
            ncols = int(header["POINTS"])
        except ValueError:
            pass

    # If we know rows and columns, reshape the data
    if nrows and ncols:
        try:
            # Check if reshaping is possible
            if main_data.shape[0] == nrows * ncols:
                main_data = main_data.reshape(nrows, ncols)
            elif main_data.shape[0] >= nrows and main_length == 1:
                # Single column data that needs reshaping
                main_data = main_data[: nrows * ncols].reshape(nrows, ncols)

            if debug:
                print(f"Reshaped data to {main_data.shape}")
        except Exception as e:
            if debug:
                print(f"Error reshaping data: {e}")

    # Create plot if requested
    if plot and main_data.size > 0:
        fig, ax = plot_special_gxf(main_data, header, debug=debug)
        return header, main_data, fig, ax

    return header, main_data


def plot_special_gxf(data_array, header, debug=False):
    """
    Create a plot adapted for the specific data format
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the plot
    if len(data_array.shape) == 2:
        # 2D data
        im = ax.imshow(data_array, cmap="viridis", origin="lower")
        cbar = fig.colorbar(im, ax=ax)
    else:
        # 1D data - plot as line
        ax.plot(data_array)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

    # Add title with file info
    title = "GXF Data"
    if header:
        extra_info = []
        for key in ["GRID", "POINTS", "ROWS"]:
            if key in header:
                extra_info.append(f"{key}: {header[key]}")
        if extra_info:
            title += " (" + ", ".join(extra_info) + ")"
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


# Add a function to help estimate grid dimensions if they're not in the header
def estimate_grid_dimensions(data_array):
    """Estimate grid dimensions for 1D data array"""
    n = len(data_array)

    # Find factors of n
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))

    # Choose the most square-like factor pair
    factors.sort(key=lambda x: abs(x[0] - x[1]))
    rows, cols = factors[0]

    return rows, cols


FILE = "../xtgeo-testdata/surfaces/etc/fdata_test_edit.gxf"
# Example usage:
header, data, fig, ax = read_gxf(FILE, plot=True, debug=True)
plt.show()  # To display the plot
fig.savefig("output.png")  # To save the plot
