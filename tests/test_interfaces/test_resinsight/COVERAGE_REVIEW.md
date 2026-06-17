# Review: ResInsight test-coverage commits (`b110b9d6` → `5aadbe5b`)

This document captures feedback on the six commits that raised coverage of
`src/xtgeo/interfaces/resinsight/` from ~56% to a reported 100%. Items are kept
separate so we can go through them one by one.

Commits under review:

- `5aadbe5b` — exclude unreachable rips import branch from coverage (pragma)
- `d998a94c` — cover rips_utils project/save/close/terminate wrappers
- `7e375552` — cover GridReader/GridWriter without ResInsight
- `db427f25` — cover GridDataResInsight `__eq__` and actnum default
- `f23a9096` — cover `_BaseResInsightDataRW` without ResInsight
- `b110b9d6` — cover rips_utils `require_rips()` call sites

---

## What's good

### G1. Right strategy
The core insight is sound: the module was stuck at ~56% only because almost
everything sat behind `@pytest.mark.requires_resinsight` and got skipped in CI.
Converting that into fast, dependency-free unit tests that run unconditionally
is the correct fix, and it complements (does not replace) the integration tests.

### G2. Behavioral assertions, not line-touching
Most tests assert real outcomes — `find_last` vs first match, the
replace-vs-create branch, exception wrapping, dtype preservation, caching
happening exactly once. That is meaningfully better than coverage-padding tests
that just call a function and assert nothing.

### G3. Good commit hygiene
One concern per commit, descriptive messages explaining *why* (the CI-skip
rationale), correct `TST:` prefix. Easy to review and revert individually.

### G4. Edge cases covered
Empty case list, no-match, wrong-type case, `actnum=None` default,
non-`GridDataResInsight` equality, unhashability. These are the spots real bugs
hide.

---

## Weaknesses / what could be improved

### W1. Heavy white-box coupling
Several tests reach past the public surface:

- `reader._ripsapi_utils = _FakeUtils(...)` in `test_resinsight_grid_unit.py`
  assigns a private attribute directly, bypassing `RipsApiUtils.__init__`.
- `monkeypatch.setattr(_resinsight_base, "RipsApiUtils", ...)` swaps an internal
  symbol.

These pass today, but they are pinned to implementation details (the cache
attribute name, the import location). A harmless refactor — renaming
`_ripsapi_utils` or restructuring the cache — would break tests even though
behavior is unchanged. That is the classic fragility of coverage-driven unit
tests. Not wrong, but worth knowing the trade-off.

### W2. Duplicated fakes across three files
`_FakeInstance`, `_FakeProject`, `_FakeCase` exist in slightly different shapes
in `test_rips_utils_unit.py`, `test_resinsight_base_unit.py`, and
`test_resinsight_grid_unit.py`. There is already a `conftest.py` in that folder —
consolidating the fakes there as fixtures/helpers would cut the duplication and
give one canonical fake to maintain when the real `rips` contract shifts.

### W3. The fakes can't catch contract drift — and 100% now hides that
These tests verify xtgeo's *glue logic*, not that it matches the real `rips`
API. If ResInsight changed `export_corner_point_grid()`'s return order, or
renamed `CornerPointCase`, every unit test would still pass at 100% while
production breaks. The brittle string check in the source —
`type(case).__name__.split(".")[-1] == "CornerPointCase"` — is a good example:
the fake is *named* `CornerPointCase` to satisfy it, so the test can never detect
if the real class name differs. The `requires_resinsight` tests remain the only
real contract check; the 100% number should not be read as "fully verified."

### W4. The pragma trades real signal for a green number
`# pragma: no cover` on the import branch makes the report say 100%, but those 8
lines are genuinely untested. The Step 5 alternative (`importlib.reload` with a
fake `rips` in `sys.modules`) would have given *actual* coverage of that logic —
including the symbol-missing error path, which is user-facing. The pragma is the
pragmatic choice, but it is the one change of the six that reduces, rather than
increases, what is truly exercised. If that error path matters, Step 5 is worth
revisiting.

### W5. Minor: mutable class-level counter
`_FakeRipsApiUtils.construct_count` is class state reset in the fixture. Fine
under process isolation, but an instance/closure counter would be cleaner and
immune to ordering surprises.

### W6. Minor: a couple of assertions could be tighter
`test_reader_load_builds_grid_data` checks dtypes but not that array *sizes*
match the declared dims, and `test_writer_save_creates_when_no_existing_case`
does not assert the `coord/zcorn/actnum` arrays were forwarded — only
`name`/`nx`. Low value, but cheap to strengthen.

---

## Bottom line

Good, well-structured changes that genuinely improve CI signal and lock in the
branch logic — worth merging. Two things to flag to a reviewer:

1. The 100% is partly cosmetic (pragma + fakes that can't detect contract
   drift), so don't let it create false confidence in the ResInsight
   integration.
2. The white-box hooks and duplicated fakes are a small maintenance liability
   that consolidating into `conftest.py` would mostly resolve.

Possible follow-ups:

- Consolidate fakes into `conftest.py` (addresses W2, partially W1).
- Implement the real Step 5 reload test and drop the pragma (addresses W4).
