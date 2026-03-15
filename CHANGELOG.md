# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
