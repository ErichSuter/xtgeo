"""
TODO: will probably not use these dicts
"""

from __future__ import annotations

from typing import TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

try:
    from typing import closed  # type: ignore[attr-defined]
except ImportError:
    _T = TypeVar("_T")

    def closed(cls: _T) -> _T:  # no-op fallback
        return cls


@closed
class GXFDict(TypedDict):
    ncol: int
    nrow: int
    xinc: float
    yinc: float
    xori: float
    yori: float
    rotation: float
    # The GXF spec allows DUMMY to be int or float, and it is important to preserve
    # the type for correct handling of masking.
    dummy: int | float
    values: npt.NDArray[np.float64]


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
        dummy_val = data["dummy"]
        masked_values = np.ma.masked_equal(values, dummy_val)

        return cls(
            ncol=int(data["ncol"]),
            nrow=int(data["nrow"]),
            xinc=float(data["xinc"]),
            yinc=float(data["yinc"]),
            xori=float(data["xori"]),
            yori=float(data["yori"]),
            rotation=float(data["rotation"]),
            dummy=dummy_val,
            values=masked_values,
        )
