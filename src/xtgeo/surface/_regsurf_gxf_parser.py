"""GXF file parser/writer for regular surfaces.

GXF (Grid eXchange Format) is a simple ASCII format for regular grid data.
This module provides functionality to read and write a subset of the GXF format,
only a subset of the keys are supported.

Documentation: https://pubs.usgs.gov/of/1999/of99-514/grids/gxf.pdf

Supported keys are:

- #POINTS
- #ROWS
- #PTSEPARATION
- #RWSEPARATION
- #XORIGIN
- #YORIGIN
- #ROTATION
- #DUMMY
- #GRID

# TODO: check this, then delete comment
#GRID contains values in row-major order (rows of length ncol), but XTGeo's internal regular surface representation is (ncol, nrow), therefore a transpose is applied on read/write.

Lines starting with ``!`` are comments and ignored. Lines that do not start with
``#`` are considered free text and ignored (outside the ``#GRID`` section).

Keys prefixed with ``##`` are treated as extension keys and ignored with a warning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

try:
    from typing import closed  # type: ignore[attr-defined]
except ImportError:
    _T = TypeVar("_T")

    def closed(cls: _T) -> _T:  # no-op fallback
        return cls


from xtgeo.io._file import FileWrapper

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike


@closed
class GXFDict(TypedDict):
    ncol: int
    nrow: int
    xinc: float
    yinc: float
    xori: float
    yori: float
    rotation: float
    dummy: float
    values: npt.NDArray[np.float64]


@dataclass(frozen=True)
class GXFData:
    """Internal immutable data representation for a regular surface in GXF format."""

    # GXF spec default values for optional keys.
    DEFAULTS: ClassVar[dict[str, float]] = {
        "PTSEPARATION": 1.0,
        "RWSEPARATION": 1.0,
        "XORIGIN": 0.0,
        "YORIGIN": 0.0,
        "ROTATION": 0.0,
    }

    ncol: int
    nrow: int
    xinc: float
    yinc: float
    xori: float
    yori: float
    rotation: float
    dummy: float
    values: np.ma.MaskedArray


    def __post_init__(self) -> None:
        values = np.ma.array(self.values, dtype=np.float64, copy=True)

        if values.shape != (self.ncol, self.nrow):
            raise ValueError(
                "Invalid surface values shape. "
                f"Expected {(self.ncol, self.nrow)}, got {values.shape}."
            )

        if self.ncol <= 0 or self.nrow <= 0:
            raise ValueError("ncol and nrow must be positive integers.")

        if self.xinc <= 0.0 or self.yinc <= 0.0:
            raise ValueError("xinc and yinc must be positive numbers.")

        # Ensure undefined nodes are represented by a mask.
        if values.mask is np.ma.nomask:
            values = np.ma.masked_equal(values, self.dummy)

        object.__setattr__(self, "values", values)


    @staticmethod
    def _strip_optional_quotes(value: str) -> str:
        stripped = value.strip()
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            return stripped[1:-1]
        return stripped


    @staticmethod
    def _next_value_line(
        lines: list[str], start_index: int, fileref_errmsg: str
    ) -> int:
        """Find the next non-empty line index used as a value line."""
        idx = start_index
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

        if idx >= len(lines):
            raise ValueError(f"In file {fileref_errmsg}: Missing value for key.")

        return idx


    @staticmethod
    def _format_number(value: float | int) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{value:.16g}"


    @classmethod
    def _parse_gxf(cls, text: str, fileref_errmsg: str) -> Self:
        lines = text.splitlines()

        scalar_values: dict[str, float | int] = {}
        grid_values: list[float] = []
        grid_found = False

        int_keys = {"POINTS", "ROWS"}
        float_keys = {
            "PTSEPARATION",
            "RWSEPARATION",
            "XORIGIN",
            "YORIGIN",
            "ROTATION",
            "DUMMY",
        }
        scalar_keys = int_keys | float_keys

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            if not stripped or stripped.startswith("!"):
                i += 1
                continue

            if not stripped.startswith("#"):
                # Free text outside the grid section.
                i += 1
                continue

            if stripped.startswith("##"):
                ext_key = stripped[2:].strip() or "<EMPTY>"
                i = cls._next_value_line(lines, i + 1, fileref_errmsg)
                warnings.warn(
                    f"In file {fileref_errmsg}: Ignoring unsupported extension "
                    f"key '##{ext_key}'.",
                    UserWarning,
                    stacklevel=3,
                )
                i += 1
                continue

            key = stripped[1:].strip().upper()

            if key == "GRID":
                grid_found = True
                i += 1
                while i < len(lines):
                    grid_line = lines[i].strip()
                    i += 1

                    if not grid_line or grid_line.startswith("!"):
                        continue

                    if grid_line.startswith("#"):
                        raise ValueError(
                            f"In file {fileref_errmsg}: Unexpected key '{grid_line}' "
                            "inside #GRID section."
                        )

                    for token in grid_line.split():
                        try:
                            grid_values.append(float(token))
                        except ValueError as err:
                            raise ValueError(
                                f"In file {fileref_errmsg}: Invalid value '{token}' "
                                "inside #GRID section."
                            ) from err

                break

            if key == "SENSE":
                # SENSE controls both the orientation
                # of the grid and right-handedness/left-handedness
                # of the coordinate system.
                # The current implementation does not take this parameter
                # into account.
                i = cls._next_value_line(lines, i + 1, fileref_errmsg)
                warnings.warn(
                    f"In file {fileref_errmsg}: Ignoring '#SENSE' key. "
                    "This key may affect grid orientation but is not "
                    "supported; verify that the surface orientation "
                    "and handedness is correct.",
                    UserWarning,
                    stacklevel=3,
                )
                i += 1
                continue

            if key not in scalar_keys:
                i = cls._next_value_line(lines, i + 1, fileref_errmsg)
                warnings.warn(
                    f"In file {fileref_errmsg}: Ignoring unsupported "
                    f"key '#{key}'.",
                    UserWarning,
                    stacklevel=3,
                )
                i += 1
                continue

            if key in scalar_values:
                raise ValueError(
                    f"In file {fileref_errmsg}: Duplicate key '#{key}' is not allowed."
                )

            i = cls._next_value_line(lines, i + 1, fileref_errmsg)
            raw_value = lines[i].strip()
            if raw_value.startswith("#"):
                raise ValueError(
                    f"In file {fileref_errmsg}: Missing value for key '#{key}'."
                )

            value_txt = cls._strip_optional_quotes(raw_value)
            try:
                parsed_value: float | int
                parsed_value = int(value_txt) if key in int_keys else float(value_txt)
            except ValueError as err:
                raise ValueError(
                    f"In file {fileref_errmsg}: Invalid value '{raw_value}' for "
                    f"key '#{key}'."
                ) from err

            scalar_values[key] = parsed_value
            i += 1

        # Only #POINTS, #ROWS and #GRID are required per the GXF spec.
        # Other keys have defaults; warn when a default is used.
        required = ["POINTS", "ROWS"]
        missing_required = [k for k in required if k not in scalar_values]
        if missing_required:
            raise ValueError(
                f"In file {fileref_errmsg}: Missing mandatory keys: "
                f"{missing_required}."
            )

        if not grid_found:
            raise ValueError(
                f"In file {fileref_errmsg}: Missing mandatory key '#GRID'."
            )

        # Apply spec defaults and warn for each.
        for dkey, dval in cls.DEFAULTS.items():
            if dkey not in scalar_values:
                scalar_values[dkey] = dval
                warnings.warn(
                    f"In file {fileref_errmsg}: Key '#{dkey}' not found, "
                    f"using default value {dval}.",
                    UserWarning,
                    stacklevel=3,
                )

        has_dummy = "DUMMY" in scalar_values
        if not has_dummy:
            warnings.warn(
                f"In file {fileref_errmsg}: Key '#DUMMY' not found, "
                "all grid values will be treated as valid.",
                UserWarning,
                stacklevel=3,
            )

        ncol = int(scalar_values["POINTS"])
        nrow = int(scalar_values["ROWS"])
        expected_values = ncol * nrow
        if len(grid_values) != expected_values:
            raise ValueError(
                f"In file {fileref_errmsg}: Number of values in #GRID section "
                f"is {len(grid_values)}, expected {expected_values} "
                f"(ncol*nrow = {ncol}*{nrow})."
            )

        # GXF grid values are read as rows of length ncol. XTGeo stores regular
        # surfaces as (ncol, nrow), therefore transpose from (nrow, ncol).
        values_2d = np.array(grid_values, dtype=np.float64).reshape((nrow, ncol)).T

        if has_dummy:
            dummy_val = float(scalar_values["DUMMY"])
            masked_values = np.ma.masked_equal(values_2d, dummy_val)
        else:
            # TODO: use default value (what is an internal sentinel?)
            dummy_val = 1e33  # internal sentinel; no actual masking
            masked_values = np.ma.array(values_2d)

        return cls(
            ncol=ncol,
            nrow=nrow,
            xinc=float(scalar_values["PTSEPARATION"]),
            yinc=float(scalar_values["RWSEPARATION"]),
            xori=float(scalar_values["XORIGIN"]),
            yori=float(scalar_values["YORIGIN"]),
            rotation=float(scalar_values["ROTATION"]),
            dummy=dummy_val,
            values=masked_values,
        )


    @classmethod
    def from_file(
        cls,
        file: FileLike,
        encoding: str = "utf-8",
    ) -> Self:
        """Read and parse a GXF file.

        Args:
            file: Path to GXF file or a file-like object (BytesIO or StringIO).
            encoding: Text encoding for the input file (default: 'utf-8').
        """

        wrapped_file = FileWrapper(file)
        if not wrapped_file.check_file():
            raise FileNotFoundError(
                f"In file {wrapped_file.name}: The file does not exist."
            )

        with wrapped_file.get_text_stream_read(encoding=encoding) as stream:
            return cls._parse_gxf(stream.read(), fileref_errmsg=str(wrapped_file.name))


    def to_file(
        self,
        file: FileLike,
        encoding: str = "utf-8",
    ) -> None:
        """Write GXFData to a file-like target in GXF format.

        Args:
            file: Path to GXF file or a file-like object (BytesIO or StringIO).
            encoding: Text encoding for the output file (default: 'utf-8').
        """

        wrapped_file = FileWrapper(file)

        with wrapped_file.get_text_stream_write(encoding=encoding) as stream:
            stream.write("#POINTS\n")
            stream.write(f'"{self._format_number(self.ncol)}"\n')

            stream.write("#ROWS\n")
            stream.write(f'"{self._format_number(self.nrow)}"\n')

            stream.write("#PTSEPARATION\n")
            stream.write(f'"{self._format_number(self.xinc)}"\n')

            stream.write("#RWSEPARATION\n")
            stream.write(f'"{ self._format_number(self.yinc)}"\n')

            stream.write("#XORIGIN\n")
            stream.write(f'"{self._format_number(self.xori)}"\n')

            stream.write("#YORIGIN\n")
            stream.write(f'"{self._format_number(self.yori)}"\n')

            stream.write("#ROTATION\n")
            stream.write(f'"{self._format_number(self.rotation)}"\n')

            stream.write("#DUMMY\n")
            stream.write(f'"{self._format_number(self.dummy)}"\n')

            stream.write("#GRID\n")

            # Write rows of ncol values from internal (ncol, nrow) representation.
            # GXF spec requires lines <= 80 chars; rows may wrap but each new
            # row must start on a new line.
            max_line_length = 80
            values_for_write = np.ma.filled(self.values, fill_value=self.dummy).T
            for row in values_for_write:
                tokens = [self._format_number(float(v)) for v in row]
                current_line = ""
                for token in tokens:
                    candidate = (current_line + " " + token) if current_line else token
                    if len(candidate) > max_line_length and current_line:
                        stream.write(current_line + "\n")
                        current_line = token
                    else:
                        current_line = candidate
                if current_line:
                    stream.write(current_line + "\n")


    def to_dict(self) -> GXFDict:
        """Return a dictionary representation of the GXFData object."""

        return GXFDict(
            ncol=self.ncol,
            nrow=self.nrow,
            xinc=self.xinc,
            yinc=self.yinc,
            xori=self.xori,
            yori=self.yori,
            rotation=self.rotation,
            dummy=self.dummy,
            values=np.array(
                np.ma.filled(self.values, fill_value=self.dummy), copy=True
            ),
        )


    @classmethod
    def from_dict(cls, data: GXFDict) -> Self:
        """Create a GXFData object from a dictionary representation."""

        required = {
            "ncol",
            "nrow",
            "xinc",
            "yinc",
            "xori",
            "yori",
            "rotation",
            "dummy",
            "values",
        }
        missing = sorted(required.difference(set(data.keys())))
        if missing:
            raise ValueError(f"Missing mandatory dictionary keys: {missing}.")

        values = np.array(data["values"], dtype=np.float64, copy=True)
        masked_values = np.ma.masked_equal(values, float(data["dummy"]))

        return cls(
            ncol=int(data["ncol"]),
            nrow=int(data["nrow"]),
            xinc=float(data["xinc"]),
            yinc=float(data["yinc"]),
            xori=float(data["xori"]),
            yori=float(data["yori"]),
            rotation=float(data["rotation"]),
            dummy=float(data["dummy"]),
            values=masked_values,
        )
