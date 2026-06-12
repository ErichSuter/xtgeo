"""Unit tests for rips_utils that don't require a running ResInsight.

These exercise the call sites of :func:`require_rips` and the
``rips.Instance`` lookups by injecting a fake ``rips`` module into both
``_rips_package`` and ``rips_utils``. Their purpose is to lift patch coverage
on the require-rips guard lines that the integration tests in
``test_rips_utils.py`` cannot reach without a real ResInsight install.
"""

from __future__ import annotations

import types

import pytest

from xtgeo.interfaces.resinsight import _rips_package, rips_utils
from xtgeo.interfaces.resinsight.rips_utils import RipsApiUtils


class _FakeInstance:
    """Stand-in for a real ``rips.Instance``."""

    def __init__(self, location: str = "localhost:50051") -> None:
        self.location = location

    def exit(self) -> None:  # used by RipsApiUtils.terminate
        pass


def _make_fake_rips_module(
    *,
    find_result: _FakeInstance | None = None,
    find_raises: Exception | None = None,
    port_ctor_raises: Exception | None = None,
    launch_result: _FakeInstance | None = None,
) -> types.ModuleType:
    """Build a minimal fake ``rips`` module with the symbols xtgeo touches."""
    mod = types.ModuleType("rips")

    class Instance(_FakeInstance):
        def __init__(self, port: int | None = None) -> None:
            if port_ctor_raises is not None:
                raise port_ctor_raises
            super().__init__(location=f"localhost:{port}")

        @staticmethod
        def find() -> _FakeInstance:
            if find_raises is not None:
                raise find_raises
            assert find_result is not None
            return find_result

        @staticmethod
        def launch(**_: object) -> _FakeInstance | None:
            return launch_result

    mod.Instance = Instance  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def install_fake_rips(monkeypatch):
    """Install a fake rips module on both modules and stub version()
    so that ``_check_rips_version()`` passes.
    """

    def _install(fake: types.ModuleType) -> None:
        monkeypatch.setattr(_rips_package, "rips", fake)
        monkeypatch.setattr(rips_utils, "rips", fake)
        monkeypatch.setattr(
            _rips_package, "version", lambda _name: _rips_package.MIN_RIPS_VERSION
        )

    return _install


# --- RipsApiUtils.__init__ ----------------------------------------------------


def test_init_with_none_calls_find_instance(install_fake_rips):
    found = _FakeInstance(location="localhost:50051")
    install_fake_rips(_make_fake_rips_module(find_result=found))

    util = RipsApiUtils(instance_or_port=None)

    assert util.instance is found


def test_init_with_int_port_calls_instance_ctor(install_fake_rips):
    install_fake_rips(_make_fake_rips_module())

    util = RipsApiUtils(instance_or_port=51234)

    assert util.instance.location == "localhost:51234"


def test_init_with_existing_instance_assigns_directly(install_fake_rips):
    fake = _make_fake_rips_module()
    install_fake_rips(fake)
    existing = fake.Instance(port=50099)  # type: ignore[attr-defined]

    util = RipsApiUtils(instance_or_port=existing)

    assert util.instance is existing


def test_init_with_invalid_type_raises(install_fake_rips):
    install_fake_rips(_make_fake_rips_module())

    with pytest.raises(TypeError, match="instance_or_port must be"):
        RipsApiUtils(instance_or_port="not a valid type")  # type: ignore[arg-type]


def test_init_raises_when_rips_missing(monkeypatch):
    monkeypatch.setattr(_rips_package, "rips", None)
    monkeypatch.setattr(_rips_package, "_rips_import_error", None)
    monkeypatch.setattr(rips_utils, "rips", None)

    with pytest.raises(RuntimeError, match="not available"):
        RipsApiUtils(instance_or_port=None)


# --- RipsApiUtils.launch_instance --------------------------------------------


def test_launch_instance_returns_instance(install_fake_rips):
    launched = _FakeInstance(location="launched:50051")
    install_fake_rips(_make_fake_rips_module(launch_result=launched))

    result = RipsApiUtils.launch_instance(executable="/bin/true")

    assert result is launched


def test_launch_instance_raises_when_launch_returns_none(install_fake_rips):
    install_fake_rips(_make_fake_rips_module(launch_result=None))

    with pytest.raises(RuntimeError, match="Failed to launch"):
        RipsApiUtils.launch_instance(executable="/bin/true")


def test_launch_instance_raises_when_rips_missing(monkeypatch):
    monkeypatch.setattr(_rips_package, "rips", None)
    monkeypatch.setattr(_rips_package, "_rips_import_error", None)
    monkeypatch.setattr(rips_utils, "rips", None)

    with pytest.raises(RuntimeError, match="not available"):
        RipsApiUtils.launch_instance()


# --- RipsApiUtils.find_instance ----------------------------------------------


def test_find_instance_no_port_returns_found(install_fake_rips):
    found = _FakeInstance()
    install_fake_rips(_make_fake_rips_module(find_result=found))

    assert RipsApiUtils.find_instance(port=None) is found


def test_find_instance_no_port_wraps_exception(install_fake_rips):
    install_fake_rips(_make_fake_rips_module(find_raises=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="Unable to connect to a running"):
        RipsApiUtils.find_instance(port=None)


def test_find_instance_with_port_constructs_instance(install_fake_rips):
    install_fake_rips(_make_fake_rips_module())

    inst = RipsApiUtils.find_instance(port=50099)

    assert inst.location == "localhost:50099"


def test_find_instance_with_port_wraps_exception(install_fake_rips):
    install_fake_rips(
        _make_fake_rips_module(port_ctor_raises=RuntimeError("denied"))
    )

    with pytest.raises(RuntimeError, match="Unable to connect to a ResInsight .* 1234"):
        RipsApiUtils.find_instance(port=1234)
