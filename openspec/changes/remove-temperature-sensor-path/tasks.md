## 1. Constants and Schema

- [x] 1.1 Remove `CONF_SENSOR_TEMPERATURE`, `CONF_SENSOR_TEMPERATURE_HIGH`, `CONF_SENSOR_WIND`, `CONF_SENSOR_RAIN`, `CONF_SENSOR_RADIATION` from `const.py`
- [x] 1.2 Remove the same constants from `__init__.py` imports and the voluptuous schema; make `weather_entity` a required key (`vol.Required`)
- [x] 1.3 Verify `CONF_SENSOR_HUMIDITY` is retained in both files (optional humidity override)

## 2. Sensor Update Logic

- [x] 2.1 In `sensor.py:_async_update()`, delete the `else` branch (individual-sensor path) of the `if weather_entity_id:` block
- [x] 2.2 In `async_added_to_hass()`, remove listener registration for `CONF_SENSOR_TEMPERATURE`, `CONF_SENSOR_TEMPERATURE_HIGH`, `CONF_SENSOR_WIND`, `CONF_SENSOR_RAIN`, `CONF_SENSOR_RADIATION`
- [x] 2.3 Remove the `_get_sensor_value` calls for the removed keys that remain after the branch deletion

## 3. Config Flow

- [x] 3.1 Check `config_flow.py` for references to the removed `CONF_SENSOR_*` keys and remove them
- [x] 3.2 Make `weather_entity` a required field in the config-flow UI (if it isn't already)

## 4. Example Configuration and Docs

- [x] 4.1 Update `example_configuration.yaml` to remove the individual-sensor stanza; ensure only `weather_entity` + optional `sensor_humidity` appear in the weather section
- [x] 4.2 Update `README.md` to reflect the single weather-entity input path; remove sensor-path documentation
- [x] 4.3 Add a breaking-change entry to `CHANGELOG.md` with migration instructions

## 5. Validation

- [ ] 5.1 Run `ha-test/restart.sh` and confirm the integration starts without errors
- [ ] 5.2 Run `HA_TOKEN=$(cat .ha_token) ha-test/validate.sh` and confirm exit 0
- [ ] 5.3 Manually verify that removing `weather_entity` from configuration.yaml produces a HA validation error at startup (not a silent runtime failure)
<!-- Docker Desktop was not running during implementation; tasks 5.1-5.3 require manual validation once Docker is available -->
