## Context

`sensor.py:_async_update()` currently branches on `if weather_entity_id:` / `else`. The `else` branch reads up to six individual HA sensor entities (`sensor_temperature`, `sensor_temperature_high`, `sensor_humidity`, `sensor_wind`, `sensor_rain`, `sensor_radiation`) and assembles `Weather` objects directly from their states. The `if` branch fetches an hourly forecast from a `weather_entity`, derives apparent temperature (BOM formula) for the cold extreme, and takes the raw max as the warm extreme.

The weather-entity path has been the documented primary path since the forecast feature landed. No known users depend solely on the individual-sensor path without a weather entity.

## Goals / Non-Goals

**Goals:**
- Delete the `else` branch and all sensor-entity plumbing that feeds it.
- Remove the associated `CONF_SENSOR_TEMPERATURE`, `CONF_SENSOR_TEMPERATURE_HIGH`, `CONF_SENSOR_WIND`, `CONF_SENSOR_RAIN`, and `CONF_SENSOR_RADIATION` constants and schema keys.
- Leave the `person` config block (`CONF_SENSOR_ACTIVITY`, `CONF_SENSOR_AGE`, `CONF_SENSOR_GENDER`) and optional humidity sensor completely intact.
- Keep `CONF_SENSOR_HUMIDITY` — it is currently usable alongside a weather entity to override forecast humidity with a local sensor, so it stays.
- Make `weather_entity` required in the schema (currently optional).

**Non-Goals:**
- Changing the forecast extraction logic (`_extract_weather_from_forecast`).
- Any changes to `engine.py`, `main.py`, or CLI.
- Adding a migration assistant or automatic config conversion.

## Decisions

### Make `weather_entity` required (not just remove the alternative)

Removing the `else` branch without making `weather_entity` required would leave a configuration state that produces no weather at all (silent fallback to 20 °C). Requiring `weather_entity` in the voluptuous schema makes the breakage explicit at startup rather than silently wrong at runtime.

_Alternative considered: keep `weather_entity` optional and log a hard error when absent._ Rejected — voluptuous validation at startup is the HA convention and gives a clear, actionable error message.

### Retain `CONF_SENSOR_HUMIDITY` as an optional override

Humidity affects apparent temperature and PMV. A local humidity sensor is more accurate than forecast humidity. Keeping the key preserves useful precision without adding complexity.

_Alternative: remove it for simplicity._ Rejected — the override is already wired and tested; removing it loses capability with no simplicity gain.

### No deprecation period

This is a pre-1.0 custom integration. A one-step removal is acceptable. The CHANGELOG will document the breaking change clearly so users can migrate.

## Risks / Trade-offs

- **Breaking change for sensor-only users** → Mitigation: clear CHANGELOG entry with migration instructions (add a `weather_entity` line to configuration.yaml).
- **`config_flow.py` may also reference removed keys** → Check and update UI flow to drop the removed fields; they are not part of the primary config-flow path but could be referenced.
- **HA restart required after config change** → Expected HA behaviour; no additional mitigation needed.

## Migration Plan

1. User adds `weather_entity: weather.<provider>` to their configuration.yaml.
2. Remove the old `weather_sensors` keys (`sensor_temperature` etc.).
3. Restart Home Assistant.
4. Validate via `ha-test/validate.sh`.

No rollback path needed — users can pin to the previous release tag if required.

## Open Questions

_(none — scope is well-defined)_
