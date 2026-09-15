## Why

The integration supports two input paths: a modern `weather_entity` path that pulls an hourly forecast and a legacy individual-sensor path (`sensor_temperature`, `sensor_temperature_high`, etc.). The dual paths add dead-weight complexity to both configuration docs and the sensor update logic. Now that the weather-entity path is stable and covers all use cases, the old sensor path should be removed so there is a single, coherent way to configure the integration.

## What Changes

- **BREAKING**: Remove `sensor_temperature`, `sensor_temperature_high`, `sensor_humidity`, `sensor_wind`, `sensor_rain`, and `sensor_radiation` configuration keys from `weather_sensors`.
- Remove the individual-sensor code branch in `_async_update()` that reads those entities.
- Remove the `CONF_SENSOR_TEMPERATURE`, `CONF_SENSOR_TEMPERATURE_HIGH`, `CONF_SENSOR_WIND`, `CONF_SENSOR_RAIN`, `CONF_SENSOR_RADIATION` constants (keep `CONF_SENSOR_HUMIDITY` only if humidity is still optionally overridden alongside a weather entity; otherwise remove it too).
- Update `__init__.py` schema validation to drop the removed keys.
- Update `example_configuration.yaml`, `README.md`, and `CHANGELOG.md` to reflect the removed options.
- Keep `CONF_SENSOR_ACTIVITY`, `CONF_SENSOR_AGE`, and `CONF_SENSOR_GENDER` — these belong to the `person` config block and are unaffected.

## Capabilities

### New Capabilities

_(none — this is a removal change)_

### Modified Capabilities

- `weather-forecast-input`: The weather-entity path becomes the **only** supported weather input method; the alternative sensor path is removed from the spec.

## Impact

- **`custom_components/kleidungsempfehlung/sensor.py`**: Delete the `else` branch of the `if weather_entity_id:` block in `_async_update()` and the listener registration for the now-removed sensor keys.
- **`custom_components/kleidungsempfehlung/__init__.py`**: Remove removed `CONF_SENSOR_*` keys from the voluptuous schema.
- **`custom_components/kleidungsempfehlung/const.py`**: Remove unused `CONF_SENSOR_*` constants.
- **`example_configuration.yaml`**: Remove the individual-sensor example stanza.
- **`README.md` / `CHANGELOG.md`**: Update accordingly.
- No changes to `engine.py`, `main.py`, or the ILP/heuristic solvers.
- Users who configured the integration with individual sensors only (no `weather_entity`) will need to migrate.
