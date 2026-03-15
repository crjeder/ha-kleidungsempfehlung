## Context

Two identical copies of the engine exist at:
- `engine.py` (repo root, ~600 lines) — imported by `main.py` CLI
- `custom_components/kleidungsempfehlung/engine.py` — imported by HA sensor via relative import `from .engine import ...`

A `diff` of the two files produces no output — they are currently byte-for-byte identical. The HA component uses only relative imports internally, so it has no dependency on the root file.

## Goals / Non-Goals

**Goals:**
- Single engine source: `custom_components/kleidungsempfehlung/engine.py` is the canonical file
- Root `engine.py` becomes a re-export shim (3 lines)
- `main.py` continues to work without modification
- HA integration continues to work without modification
- No behavior change of any kind

**Non-Goals:**
- Changing the engine logic
- Changing the CLI argument interface
- Changing the HA YAML config format
- Unifying CLI and HA config formats (deferred)

## Decisions

### D1: HA component engine is canonical (not root)

**Decision**: `custom_components/kleidungsempfehlung/engine.py` is the source of truth.

**Rationale**: The HA component uses relative imports (`from .engine import ...`), so it can only ever point to files within its own package. The root `engine.py` has no such constraint — it can `import` from anywhere. This asymmetry means only the root can be the shim, not the HA file.

**Alternative considered**: Making root canonical and having HA component import from `....engine` — rejected because Python relative imports cannot traverse above the package root, and HA's custom component loader may restrict sys.path further.

### D2: Use `from ... import *` (star import) in the shim

**Decision**: Root `engine.py` contains:
```python
# engine.py — re-export shim
# The canonical engine lives in the HA component package.
from custom_components.kleidungsempfehlung.engine import *
```

**Rationale**: `main.py` imports specific names (`SmartClothingEngine`, `Weather`, `SLOTS`, `get_item_id`, `is_locked`, `make_entry`). A star import re-exports all public names, so `main.py` continues to work with zero changes. No `__all__` is needed since the HA engine has no leading-underscore private symbols that need hiding.

**Alternative considered**: Explicit re-exports (`from custom_components.kleidungsempfehlung.engine import SmartClothingEngine, Weather, ...`) — more explicit but brittle: any new export added to the engine would require updating the shim too.

## Risks / Trade-offs

- **Python path dependency**: `from custom_components.kleidungsempfehlung.engine import *` requires `custom_components/` to be importable. This works when running `python main.py` from the repo root (the default), but breaks if a user runs main.py from a different working directory. → _Mitigation_: This is an existing constraint — `main.py` already assumes it's run from the repo root (it constructs paths via `Path(__file__).parent`). No regression.

- **IDE / linter confusion**: Some IDEs may not resolve star imports and will show false "undefined name" warnings for names imported from the shim. → _Mitigation_: Cosmetic only; runtime behavior is correct.

- **`__all__` caveat**: If the HA engine ever adds `__all__`, the star import will respect it and only export listed names. If a name is accidentally omitted from `__all__`, `main.py` would break. → _Mitigation_: Low probability; the engine has no current `__all__` and no plans to add one.

## Migration Plan

1. Replace `engine.py` (root) content with the 3-line shim
2. Verify `python main.py --temp 10 --wind 2 --met 1.2` still runs correctly
3. Verify HA integration tests still pass (`./ha-test/restart.sh` + `./ha-test/validate.sh`)
4. Commit

No rollback strategy needed — the change is trivially reversible by restoring the original engine content.

## Open Questions

None. The approach is unambiguous.
