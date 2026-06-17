# Mocks vs MonkeyPatch

`Mock` and `monkeypatch` both help tests replace real behavior with controlled behavior, but they operate at different levels.

Short version:

- `Mock` is a fake object.
- `monkeypatch` is a temporary replacement mechanism.

They are often used together: create a fake or mock object, then use `monkeypatch` to install it where the production code will look for it.

## Mocks

A mock is an object that stands in for a real dependency. In Python this usually means `unittest.mock.Mock`, `MagicMock`, or `patch`.

Mocks are useful when you want to:

- Replace an external dependency.
- Return controlled values.
- Raise controlled exceptions.
- Record how something was called.
- Assert interactions.

Example:

```python
from unittest.mock import Mock

fake_project = Mock()
fake_project.save.return_value = None

util.save_project("project.rsp")

fake_project.save.assert_called_once_with("project.rsp")
```

The mock itself carries behavior and call history.

## MonkeyPatch

`monkeypatch` is pytest's fixture for temporarily changing program state during a test. It can replace attributes, environment variables, dictionary values, paths, or working directories. After the test, pytest automatically restores the original state.

Example:

```python
def test_uses_fake_rips(monkeypatch):
    monkeypatch.setattr(rips_utils, "rips", fake_rips_module)
```

Here, `monkeypatch` is not the fake. It is the tool that installs `fake_rips_module` into `rips_utils`.

## Similarities

Both are used to isolate the code under test.

Both can prevent tests from touching real systems such as:

- Filesystems.
- Network APIs.
- Databases.
- GUI applications.
- External Python packages.
- Environment-dependent configuration.

Both are commonly used to make tests deterministic.

## Differences

| Aspect | Mock | MonkeyPatch |
|---|---|---|
| What it is | A fake object with configurable behavior | A pytest fixture for temporary replacement |
| Main job | Simulate and record behavior | Swap values in and out safely during a test |
| Tracks calls? | Yes | No |
| Automatically restores state? | Not by itself, unless used via `patch` context/decorator | Yes |
| Typical source | `unittest.mock` | `pytest` |
| Replaces attributes? | Can, often via `patch` | Yes, directly |
| Best for | Interaction assertions | Replacing global, module, or environment state |

## Common Usage Patterns

Use a mock when you care about calls:

```python
from unittest.mock import Mock

service = Mock()
service.fetch.return_value = {"status": "ok"}

result = service.fetch("abc")

service.fetch.assert_called_once_with("abc")
assert result == {"status": "ok"}
```

Use `monkeypatch` when code imports or references something globally:

```python
def test_env_mode(monkeypatch):
    monkeypatch.setenv("APP_MODE", "test")

    assert get_mode() == "test"
```

Use them together when production code looks up a dependency from a module:

```python
from unittest.mock import Mock


def test_save_project(monkeypatch):
    fake_rips = Mock()
    monkeypatch.setattr(rips_utils, "rips", fake_rips)

    # Code under test now sees fake_rips instead of real rips.
```

## In The ResInsight Tests

The fake classes in `tests/test_interfaces/test_resinsight/conftest.py` are mostly hand-written fakes, not `Mock` objects.

For example, `install_fake_rips` uses `monkeypatch` to replace module-level references:

```python
monkeypatch.setattr(_rips_package, "rips", fake)
monkeypatch.setattr(rips_utils, "rips", fake)
```

That means tests can exercise `RipsApiUtils` without importing the real `rips` package or starting ResInsight.

The pattern is:

1. Build a fake `rips` module.
2. Use `monkeypatch` to install it.
3. Run `RipsApiUtils`.
4. Assert the fake recorded the expected behavior.

## Rule Of Thumb

Use `Mock` when you want a programmable fake with call assertions.

Use `monkeypatch` when you need to temporarily change where the code gets a dependency from.

Use hand-written fakes when behavior matters enough that a named fake class is clearer than a highly configured mock.
