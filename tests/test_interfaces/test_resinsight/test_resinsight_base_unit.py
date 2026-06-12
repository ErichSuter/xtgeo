"""Unit tests for _BaseResInsightDataRW that don't require a running ResInsight.

These inject a fake ``RipsApiUtils`` into ``_resinsight_base`` so the base
class logic (caching, instance/project/case lookup) can be exercised without
a real ResInsight install. The integration tests that use a live instance
live in ``test_resinsight_grid.py`` and are gated on ``requires_resinsight``.

The fake objects are defined once in ``conftest.py`` and provided through the
``patch_resinsight_utils`` and ``ri_fakes`` fixtures.
"""

from __future__ import annotations

from xtgeo.interfaces.resinsight._resinsight_base import _BaseResInsightDataRW


def _make_rw(instance_or_port=None) -> _BaseResInsightDataRW:
    return _BaseResInsightDataRW(instance_or_port)


# --- get_ripsapi_utils caching ------------------------------------------------


def test_get_ripsapi_utils_constructs_once_and_caches(patch_resinsight_utils):
    rw = _make_rw(instance_or_port=50051)

    first = rw.get_ripsapi_utils()
    second = rw.get_ripsapi_utils()

    assert first is second, "Should cache and return the same RipsApiUtils"
    assert patch_resinsight_utils.construct_count == 1, (
        "Should construct RipsApiUtils only once"
    )
    assert first.instance_or_port == 50051


# --- get_instance / get_project ----------------------------------------------


def test_get_instance_returns_utils_instance(patch_resinsight_utils):
    rw = _make_rw()
    utils = rw.get_ripsapi_utils()

    assert rw.get_instance() is utils.instance


def test_get_project_returns_utils_project(patch_resinsight_utils):
    rw = _make_rw()
    utils = rw.get_ripsapi_utils()

    assert rw.get_project() is utils.project


# --- get_case ----------------------------------------------------------------


def _install_cases(rw: _BaseResInsightDataRW, ri_fakes, cases) -> None:
    """Force the cached utils to expose a project with the given cases."""
    utils = rw.get_ripsapi_utils()
    utils.instance.project = ri_fakes.Project(cases)


def test_get_case_returns_none_when_no_cases(patch_resinsight_utils, ri_fakes):
    rw = _make_rw()
    _install_cases(rw, ri_fakes, [])

    assert rw.get_case("ANY") is None


def test_get_case_returns_none_when_no_match(patch_resinsight_utils, ri_fakes):
    rw = _make_rw()
    _install_cases(rw, ri_fakes, [ri_fakes.Case("A"), ri_fakes.Case("B")])

    assert rw.get_case("MISSING") is None


def test_get_case_find_last_returns_last_match(patch_resinsight_utils, ri_fakes):
    rw = _make_rw()
    first = ri_fakes.Case("DUP")
    last = ri_fakes.Case("DUP")
    _install_cases(rw, ri_fakes, [first, ri_fakes.Case("OTHER"), last])

    selected = rw.get_case("DUP", find_last=True)

    assert selected is last


def test_get_case_find_first_returns_first_match(patch_resinsight_utils, ri_fakes):
    rw = _make_rw()
    first = ri_fakes.Case("DUP")
    last = ri_fakes.Case("DUP")
    _install_cases(rw, ri_fakes, [first, ri_fakes.Case("OTHER"), last])

    selected = rw.get_case("DUP", find_last=False)

    assert selected is first
