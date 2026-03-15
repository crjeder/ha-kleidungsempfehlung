## Why

The engine logic (`SmartClothingEngine`, `Weather`, helpers) exists in two identical copies — `engine.py` at the repo root and `custom_components/kleidungsempfehlung/engine.py`. Any future change must be applied twice, making divergence inevitable. The standalone root copy is no longer actively used (CLI is secondary), so this is the right time to eliminate the duplication.

## What Changes

- `custom_components/kleidungsempfehlung/engine.py` becomes the single source of truth for all engine logic — no changes to its content.
- `engine.py` (root) is replaced with a thin re-export shim: `from custom_components.kleidungsempfehlung.engine import *`
- `main.py` (CLI) continues to work unchanged — it imports from the root `engine.py`, which now re-exports everything transparently.

## Capabilities

### New Capabilities

- `engine-single-source`: One canonical engine file; root `engine.py` is a re-export shim pointing to the HA component's engine.

### Modified Capabilities

<!-- No spec-level behavior changes. This is a structural/import-path change only. -->

## Impact

- **`engine.py` (root)**: Replaced entirely — ~600 lines → 3 lines.
- **`main.py`**: No changes needed; imports continue to resolve via the shim.
- **`custom_components/kleidungsempfehlung/`**: Zero changes — uses relative imports throughout, unaffected.
- **Dependencies**: No new dependencies. `pythermalcomfort`, `scipy`, `numpy` remain in the HA component's engine.
- **Tests**: HA integration tests unaffected. CLI (`main.py`) continues to work.
