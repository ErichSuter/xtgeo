"""Shared fixtures for testing the ResInsight interface.

Shall only depend on the ResInsight interface layer
(xtgeo.interfaces.resinsight), not on other xtgeo modules."""

import logging
import pathlib
import types
from types import SimpleNamespace

import numpy as np
import pytest

from xtgeo.interfaces.resinsight import _resinsight_base, _rips_package, rips_utils
from xtgeo.interfaces.resinsight._rips_package import RipsInstanceType
from xtgeo.interfaces.resinsight.rips_utils import RipsApiUtils

DROGON_GRID = pathlib.Path("3dgrids/drogon/2/geogrid.roff")
EMERALD_GRID = pathlib.Path("3dgrids/eme/1/emerald.roff")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@pytest.fixture(scope="module")
def resinsight_instance(testdata_path) -> RipsInstanceType:
    """Create a ResInsight instance for testing.

    Use console_mode to avoid opening the full GUI, which is not needed for API tests
    and can cause issues in headless environments.
    """
    pytest.importorskip(
        "rips", reason="ResInsight API tests require 'rips' package to be installed"
    )
    logger.info("Creating ResInsight instance for testing")
    try:
        instance = RipsApiUtils.launch_instance(executable="", console_mode=True)
    except (RuntimeError, OSError) as e:
        pytest.skip(
            f"ResInsight executable not available (set RESINSIGHT_EXECUTABLE env var "
            f"or add ResInsight to PATH): {e}"
        )
    logger.info(
        "Version: ResInsight %s, RIPS %s",
        instance.version_string(),
        instance.client_version_string(),
    )
    path = pathlib.Path(testdata_path)

    drogon = instance.project.load_case(path=str(path / DROGON_GRID))
    emerald = instance.project.load_case(path=str(path / EMERALD_GRID))

    # Give the cases with same name to test 'find_last' functionality
    # in GridReader/GridWriter
    drogon.name = "EXAMPLE"
    drogon.update()
    emerald.name = "EXAMPLE"
    emerald.update()

    cases = [case.name for case in instance.project.cases()]
    logger.info("ResInsight instance created and test cases loaded %s", cases)
    yield instance

    # Teardown: close the instance after tests are done
    logger.info("Closing ResInsight instance after testing")
    instance.exit()


# ---------------------------------------------------------------------------
# Shared fakes for unit tests that do NOT require a running ResInsight.
#
# These are the single canonical definitions of the fake rips objects used by
# the *_unit.py test modules. They are exposed to tests through fixtures
# (``ri_fakes``, ``install_fake_rips``, ``patch_resinsight_utils``) rather than
# imported directly, since the test directories are not import packages.
# ---------------------------------------------------------------------------

# Dimensions used by fake corner-point grid exports (small 2x2x2 grid).
FAKE_NX = FAKE_NY = FAKE_NZ = 2


def fake_grid_arrays():
    """Return (zcorn, coord, actnum) flat arrays valid for a 2x2x2 grid."""
    zcorn = np.zeros(FAKE_NX * FAKE_NY * FAKE_NZ * 8, dtype=np.float32)
    coord = np.zeros((FAKE_NX + 1) * (FAKE_NY + 1) * 6, dtype=np.float64)
    actnum = np.ones(FAKE_NX * FAKE_NY * FAKE_NZ, dtype=np.int32)
    return zcorn, coord, actnum


class FakeProject:
    """Unified fake of a rips project.

    Supports the full surface used across the unit tests: case listing,
    corner-point grid creation (with optional failure), and save/close.
    """

    def __init__(self, cases=None, create_result=None, create_raises=None) -> None:
        self._cases = list(cases or [])
        self._create_result = create_result
        self._create_raises = create_raises
        self.created_kwargs: dict | None = None
        self.saved_with: str | None = None
        self.closed = False

    def cases(self):
        return self._cases

    def create_corner_point_grid(self, **kwargs):
        if self._create_raises is not None:
            raise self._create_raises
        self.created_kwargs = kwargs
        return self._create_result

    def save(self, name: str) -> None:
        self.saved_with = name

    def close(self) -> None:
        self.closed = True


class FakeInstance:
    """Unified fake of a ``rips.Instance``."""

    def __init__(self, location: str = "localhost:50051", project=None) -> None:
        self.location = location
        self.project = project
        self.exited = False

    def exit(self) -> None:
        self.exited = True


class FakeCase:
    """Fake case exposing ``name``, ``file_path`` and the grid export API."""

    def __init__(self, name: str, file_path: str = "src.roff") -> None:
        self.name = name
        self.file_path = file_path

    def export_corner_point_grid(self):
        zcorn, coord, actnum = fake_grid_arrays()
        return zcorn, coord, actnum, FAKE_NX, FAKE_NY, FAKE_NZ


class CornerPointCase:
    """Fake replaceable case; the class name matters (GridWriter checks it)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.file_path = ""
        self.replaced_with: tuple | None = None
        self.updated = False

    def replace_corner_point_grid(self, nx, ny, nz, coord, zcorn, actnum):
        self.replaced_with = (nx, ny, nz)

    def update(self) -> None:
        self.updated = True


class GridFileCase:
    """Fake case of a non-replaceable type (different class name)."""

    def __init__(self, name: str) -> None:
        self.name = name


class NewCase:
    """Fake case returned by ``create_corner_point_grid``."""

    def __init__(self) -> None:
        self.file_path = ""
        self.updated = False

    def update(self) -> None:
        self.updated = True


class FakeRipsApiUtils:
    """Stand-in for ``RipsApiUtils``.

    Records how often it is constructed (for caching assertions) and exposes
    an ``instance``/``project`` pair. A specific ``project`` may be supplied
    for reader/writer tests.
    """

    construct_count = 0

    def __init__(self, instance_or_port=None, *, project=None) -> None:
        type(self).construct_count += 1
        self.instance_or_port = instance_or_port
        self.instance = FakeInstance(
            project=project if project is not None else FakeProject()
        )

    @property
    def project(self):
        return self.instance.project


def make_fake_rips_module(
    *,
    find_result=None,
    find_raises=None,
    port_ctor_raises=None,
    launch_result=None,
) -> types.ModuleType:
    """Build a minimal fake ``rips`` module with the symbols xtgeo touches."""
    mod = types.ModuleType("rips")

    class Instance(FakeInstance):
        def __init__(self, port: int | None = None) -> None:
            if port_ctor_raises is not None:
                raise port_ctor_raises
            super().__init__(location=f"localhost:{port}")

        @staticmethod
        def find():
            if find_raises is not None:
                raise find_raises
            assert find_result is not None
            return find_result

        @staticmethod
        def launch(**_: object):
            return launch_result

    mod.Instance = Instance  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def ri_fakes():
    """Namespace bundle of the canonical ResInsight fakes and helpers."""
    return SimpleNamespace(
        Project=FakeProject,
        Instance=FakeInstance,
        Case=FakeCase,
        CornerPointCase=CornerPointCase,
        GridFileCase=GridFileCase,
        NewCase=NewCase,
        RipsApiUtils=FakeRipsApiUtils,
        make_rips_module=make_fake_rips_module,
        grid_arrays=fake_grid_arrays,
        NX=FAKE_NX,
        NY=FAKE_NY,
        NZ=FAKE_NZ,
    )


@pytest.fixture
def install_fake_rips(monkeypatch):
    """Install a fake rips module on both modules and stub ``version()`` so
    that ``_check_rips_version()`` passes.
    """

    def _install(fake: types.ModuleType) -> None:
        monkeypatch.setattr(_rips_package, "rips", fake)
        monkeypatch.setattr(rips_utils, "rips", fake)
        monkeypatch.setattr(
            _rips_package, "version", lambda _name: _rips_package.MIN_RIPS_VERSION
        )

    return _install


@pytest.fixture
def patch_resinsight_utils(monkeypatch):
    """Patch ``RipsApiUtils`` in ``_resinsight_base`` with the fake and reset
    its construction counter. Returns the fake class.
    """
    FakeRipsApiUtils.construct_count = 0
    monkeypatch.setattr(_resinsight_base, "RipsApiUtils", FakeRipsApiUtils)
    return FakeRipsApiUtils
