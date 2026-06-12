"""Unit tests for _BaseResInsightDataRW that don't require a running ResInsight.

These inject a fake ``RipsApiUtils`` into ``_resinsight_base`` so the base
class logic (caching, instance/project/case lookup) can be exercised without
a real ResInsight install. The integration tests that use a live instance
live in ``test_resinsight_grid.py`` and are gated on ``requires_resinsight``.
"""

from __future__ import annotations

import pytest

from xtgeo.interfaces.resinsight import _resinsight_base
from xtgeo.interfaces.resinsight._resinsight_base import _BaseResInsightDataRW


class _FakeCase:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeProject:
    def __init__(self, cases: list[_FakeCase] | None = None) -> None:
        self._cases = cases or []

    def cases(self) -> list[_FakeCase]:
        return self._cases


class _FakeInstance:
    def __init__(self, project: _FakeProject | None = None) -> None:
        self.project = project or _FakeProject()


class _FakeRipsApiUtils:
    """Stand-in for RipsApiUtils that records how often it is constructed."""

    construct_count = 0

    def __init__(self, instance_or_port=None) -> None:
        type(self).construct_count += 1
        self.instance_or_port = instance_or_port
        self.instance = _FakeInstance()

    @property
    def project(self) -> _FakeProject:
        return self.instance.project


@pytest.fixture
def patch_utils(monkeypatch):
    """Patch RipsApiUtils in _resinsight_base with the fake, reset its counter."""
    _FakeRipsApiUtils.construct_count = 0
    monkeypatch.setattr(_resinsight_base, "RipsApiUtils", _FakeRipsApiUtils)
    return _FakeRipsApiUtils


def _make_rw(instance_or_port=None) -> _BaseResInsightDataRW:
    return _BaseResInsightDataRW(instance_or_port)


# --- get_ripsapi_utils caching ------------------------------------------------


def test_get_ripsapi_utils_constructs_once_and_caches(patch_utils):
    rw = _make_rw(instance_or_port=50051)

    first = rw.get_ripsapi_utils()
    second = rw.get_ripsapi_utils()

    assert first is second, "Should cache and return the same RipsApiUtils"
    assert patch_utils.construct_count == 1, "Should construct RipsApiUtils only once"
    assert first.instance_or_port == 50051


# --- get_instance / get_project ----------------------------------------------


def test_get_instance_returns_utils_instance(patch_utils):
    rw = _make_rw()
    utils = rw.get_ripsapi_utils()

    assert rw.get_instance() is utils.instance


def test_get_project_returns_utils_project(patch_utils):
    rw = _make_rw()
    utils = rw.get_ripsapi_utils()

    assert rw.get_project() is utils.project


# --- get_case ----------------------------------------------------------------


def _install_cases(rw: _BaseResInsightDataRW, cases: list[_FakeCase]) -> None:
    """Force the cached utils to expose a project with the given cases."""
    utils = rw.get_ripsapi_utils()
    utils.instance.project = _FakeProject(cases)


def test_get_case_returns_none_when_no_cases(patch_utils):
    rw = _make_rw()
    _install_cases(rw, [])

    assert rw.get_case("ANY") is None


def test_get_case_returns_none_when_no_match(patch_utils):
    rw = _make_rw()
    _install_cases(rw, [_FakeCase("A"), _FakeCase("B")])

    assert rw.get_case("MISSING") is None


def test_get_case_find_last_returns_last_match(patch_utils):
    rw = _make_rw()
    first = _FakeCase("DUP")
    last = _FakeCase("DUP")
    _install_cases(rw, [first, _FakeCase("OTHER"), last])

    selected = rw.get_case("DUP", find_last=True)

    assert selected is last


def test_get_case_find_first_returns_first_match(patch_utils):
    rw = _make_rw()
    first = _FakeCase("DUP")
    last = _FakeCase("DUP")
    _install_cases(rw, [first, _FakeCase("OTHER"), last])

    selected = rw.get_case("DUP", find_last=False)

    assert selected is first
