## 1. Constants and config schema

- [x] 1.1 Add `CONF_WEATHER_ENTITY = "weather_entity"` to `custom_components/kleidungsempfehlung/const.py`
- [x] 1.2 Update `__init__.py` voluptuous schema to accept the optional `weather_entity` key (string, entity-id format)

## 2. Perceived temperature helper

- [x] 2.1 Add `_apparent_temperature(t_c, rh_pct, ws_ms)` pure function to `sensor.py` implementing the BOM formula
- [x] 2.2 Add `_wind_kmh_to_ms(kmh)` conversion helper (or inline the division — keep it simple)

## 3. Forecast fetch logic

- [x] 3.1 Add `async _async_fetch_forecast(entity_id)` method to `KleidungsempfehlungSensor` that calls `weather.get_forecasts` service with `type: hourly`, catches `ServiceNotFound` and falls back to `state.attributes["forecast"]`, and returns an empty list on any error with a warning log
- [x] 3.2 Add `_extract_weather_from_forecast(forecasts)` method that iterates forecast entries, converts wind speed to m/s, computes apparent temperature per entry, and returns `(cold_t, cold_ws, hot_t)` — the min-apparent-temp's `(t, ws)` and max raw temp

## 4. Sensor update integration

- [x] 4.1 In `_async_update`, detect whether `weather_entity` is configured (from `self._config`)
- [x] 4.2 When configured: call `_async_fetch_forecast`, call `_extract_weather_from_forecast`, build cold `Weather` and optional warm `Weather` objects from the result; fall back to current state temperature+wind if forecast is empty
- [x] 4.3 Add `forecast_window` dict to `self._attributes` when a weather entity is used (`entries`, `t_min_perceived`, `t_max_raw`, `wind_at_min`)

## 5. State-change subscription

- [x] 5.1 In `async_added_to_hass`, subscribe to state-change events for `weather_entity` (when configured) using `async_track_state_change_event`, same pattern as existing sensor subscriptions

## 6. Config flow (UI)

- [x] 6.1 Add optional `weather_entity` field to `config_flow.py` schema and step form so UI-configured integrations can also use this feature

## 7. Example configuration update

- [x] 7.1 Add a commented `weather_entity` example to `custom_components/kleidungsempfehlung/example_configuration.yaml` showing usage alongside (and without) individual sensor entries

## 8. Validation and testing

- [x] 8.1 Run `./ha-test/restart.sh` and confirm the integration loads without errors using the existing config (no weather_entity set)
- [x] 8.2 Add `weather_entity: weather.stub` to the test config in `ha-test/` and confirm the sensor updates and exposes `forecast_window` attribute
- [x] 8.3 Run `HA_TOKEN=$(cat .ha_token) ./ha-test/validate.sh` and confirm exit 0
