# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2] - 2026-03-15

### Changed
- `engine.py` (root) is now a re-export shim — canonical engine logic lives exclusively in `custom_components/kleidungsempfehlung/engine.py`. Eliminates code duplication; CLI (`main.py`) continues to work unchanged.

## [0.9.1] - 2026-03-15

### Fixed
- **Heuristic solver: hardcoded item IDs in default ensemble init** — `optimize_ensemble` previously seeded the starting ensemble with hardcoded IDs (`undershirt_sleeveless`, `boxer_synthetic`, `pants_standard`, `sneakers`). Items not present in the active inventory became phantom entries: they occupied layer positions (blocking upgrades), contributed 0 CLO, and could not be replaced. This was the root cause of PMV=-2.01 with the HA test inventory at 10°C. The init now picks the lightest available item per slot/fit from the actual inventory.
- **Heuristic solver: fit violation in legs init** — `pants_standard` (fit=outer) was placed at `legs[1]` (a mid-layer position), violating the fit constraint. Outer items accumulated at the wrong layer as upgrades were applied. Fixed by placing the default outer legs item at `legs[self.max_layers - 1]`.

### Added
- `thermal_leggings` item to `inventory.json` (slot=legs, fit=mid, clo=0.40) — fills the inventory gap where the only mid-fit legs items were `shorts` (0.06) and `joggers_light` (0.25). ILP can now achieve ~3.60 CLO at -10°C (wind 5 m/s), improving PMV from -2.26 to -1.94.

## [0.9.0] - 2023-03-15

### Added
- Inline ISO 7730 Fanger PMV/PPD implementation in `engine.py` — removes the `pythermalcomfort` dependency entirely, which could not be installed in the HA container (Python 3.14 + numba/llvmlite build failure)
- `claude-progress.txt` to track session progress
- `CHANGELOG.md` (this file)

### Fixed
- `ha-test/ha-config/configuration.yaml` — replaced default HA config with the proper test config (stub sensors + kleidungsempfehlung integration)
- `ha-test/ha-config/inventory.yaml` — added missing file so `!include inventory.yaml` resolves correctly
- `.ha_token` — copied to project root where `validate.sh` expects it
- `custom_components/kleidungsempfehlung/__init__.py` — replaced removed `hass.helpers.discovery` with `homeassistant.helpers.discovery` (API change in recent HA versions)
- `ha-test/validate.sh` — log checks now use `--since <container_start>` to filter out errors from previous container runs
- `ha-test/check_sensors.py` — added UTF-8 stdout wrapper for Windows to handle emoji output characters

### Removed
- `pythermalcomfort` from `manifest.json` requirements (replaced by inline implementation)
