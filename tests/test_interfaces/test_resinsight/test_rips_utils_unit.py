"""Unit tests for rips_utils that don't require a running ResInsight.

These exercise the call sites of :func:`require_rips` and the
``rips.Instance`` lookups by injecting a fake ``rips`` module into both
``_rips_package`` and ``rips_utils``. Their purpose is to lift patch coverage
on the require-rips guard lines that the integration tests in
``test_rips_utils.py`` cannot reach without a real ResInsight install.

The fake rips objects are defined once in ``conftest.py`` and provided through
the ``ri_fakes`` and ``install_fake_rips`` fixtures.
"""

from __future__ import annotations

import pytest

from xtgeo.interfaces.resinsight import _rips_package, rips_utils
from xtgeo.interfaces.resinsight.rips_utils import RipsApiUtils

# --- RipsApiUtils.__init__ ----------------------------------------------------


def test_init_with_none_calls_find_instance(install_fake_rips, ri_fakes):
    found = ri_fakes.Instance(location="localhost:50051")
    install_fake_rips(ri_fakes.make_rips_module(find_result=found))

    util = RipsApiUtils(instance_or_port=None)

    assert util.instance is found


def test_init_with_int_port_calls_instance_ctor(install_fake_rips, ri_fakes):
    install_fake_rips(ri_fakes.make_rips_module())

    util = RipsApiUtils(instance_or_port=51234)

    assert util.instance.location == "localhost:51234"


def test_init_with_existing_instance_assigns_directly(install_fake_rips, ri_fakes):
    fake = ri_fakes.make_rips_module()
    install_fake_rips(fake)
    existing = fake.Instance(port=50099)  # type: ignore[attr-defined]

    util = RipsApiUtils(instance_or_port=existing)

    assert util.instance is existing


def test_init_with_invalid_type_raises(install_fake_rips, ri_fakes):
    install_fake_rips(ri_fakes.make_rips_module())

    with pytest.raises(TypeError, match="instance_or_port must be"):
        RipsApiUtils(instance_or_port="not a valid type")  # type: ignore[arg-type]


def test_init_raises_when_rips_missing(monkeypatch):
    monkeypatch.setattr(_rips_package, "rips", None)
    monkeypatch.setattr(_rips_package, "_rips_import_error", None)
    monkeypatch.setattr(rips_utils, "rips", None)

    with pytest.raises(RuntimeError, match="not available"):
        RipsApiUtils(instance_or_port=None)


# --- RipsApiUtils.launch_instance --------------------------------------------


def test_launch_instance_returns_instance(install_fake_rips, ri_fakes):
    launched = ri_fakes.Instance(location="launched:50051")
    install_fake_rips(ri_fakes.make_rips_module(launch_result=launched))

    result = RipsApiUtils.launch_instance(executable="/bin/true")

    assert result is launched


def test_launch_instance_raises_when_launch_returns_none(install_fake_rips, ri_fakes):
    install_fake_rips(ri_fakes.make_rips_module(launch_result=None))

    with pytest.raises(RuntimeError, match="Failed to launch"):
        RipsApiUtils.launch_instance(executable="/bin/true")


def test_launch_instance_raises_when_rips_missing(monkeypatch):
    monkeypatch.setattr(_rips_package, "rips", None)
    monkeypatch.setattr(_rips_package, "_rips_import_error", None)
    monkeypatch.setattr(rips_utils, "rips", None)

    with pytest.raises(RuntimeError, match="not available"):
        RipsApiUtils.launch_instance()


# --- RipsApiUtils.find_instance ----------------------------------------------


def test_find_instance_no_port_returns_found(install_fake_rips, ri_fakes):
    found = ri_fakes.Instance()
    install_fake_rips(ri_fakes.make_rips_module(find_result=found))

    assert RipsApiUtils.find_instance(port=None) is found


def test_find_instance_no_port_wraps_exception(install_fake_rips, ri_fakes):
    install_fake_rips(ri_fakes.make_rips_module(find_raises=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="Unable to connect to a running"):
        RipsApiUtils.find_instance(port=None)


def test_find_instance_with_port_constructs_instance(install_fake_rips, ri_fakes):
    install_fake_rips(ri_fakes.make_rips_module())

    inst = RipsApiUtils.find_instance(port=50099)

    assert inst.location == "localhost:50099"


def test_find_instance_with_port_wraps_exception(install_fake_rips, ri_fakes):
    install_fake_rips(
        ri_fakes.make_rips_module(port_ctor_raises=RuntimeError("denied"))
    )

    with pytest.raises(RuntimeError, match="Unable to connect to a ResInsight .* 1234"):
        RipsApiUtils.find_instance(port=1234)


# --- project / save_project / close_project / terminate ----------------------


def _util_with_project(install_fake_rips, ri_fakes):
    """Build a RipsApiUtils wrapping a fake instance with a recording project."""
    fake = ri_fakes.make_rips_module()
    install_fake_rips(fake)
    instance = fake.Instance(port=50051)  # type: ignore[attr-defined]
    project = ri_fakes.Project()
    instance.project = project
    util = RipsApiUtils(instance_or_port=instance)
    return util, instance, project


def test_project_property_returns_instance_project(install_fake_rips, ri_fakes):
    util, _instance, project = _util_with_project(install_fake_rips, ri_fakes)

    assert util.project is project


def test_save_project_passes_name(install_fake_rips, ri_fakes):
    util, _instance, project = _util_with_project(install_fake_rips, ri_fakes)

    util.save_project("my_project.rsp")

    assert project.saved_with == "my_project.rsp"


def test_save_project_default_name_is_empty(install_fake_rips, ri_fakes):
    util, _instance, project = _util_with_project(install_fake_rips, ri_fakes)

    util.save_project()

    assert project.saved_with == ""


def test_close_project_calls_close(install_fake_rips, ri_fakes):
    util, _instance, project = _util_with_project(install_fake_rips, ri_fakes)

    util.close_project()

    assert project.closed is True


def test_terminate_calls_instance_exit(install_fake_rips, ri_fakes):
    util, instance, _project = _util_with_project(install_fake_rips, ri_fakes)

    util.terminate()

    assert instance.exited is True
