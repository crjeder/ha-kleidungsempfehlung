# TODO

## Testing
- [ ] Add multi-scenario validation (different weather/activity inputs) to validate.sh or check_sensors.py
- [ ] Verify `sensor_gender` and `sensor_age` stubs are properly wired up in sensor.py

## Engine
- [ ] Investigate why PMV=-2.01 at 10°C with full winter outfit (heuristic solver may be underperforming vs ILP)
- [ ] Add unit tests for `_pmv_ppd_fanger()` against reference values from ISO 7730 Annex D

## Integration
- [ ] Verify config_flow.py (UI-based setup) still works after __init__.py changes
