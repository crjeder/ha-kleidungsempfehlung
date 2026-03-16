## 1. Constants and Schema

- [x] 1.1 Add `CONF_PERSON_ENTITY = "person_entity"` to `const.py`
- [x] 1.2 Add `vol.Optional(CONF_PERSON_ENTITY): cv.entity_id` to the top-level `CONFIG_SCHEMA` in `__init__.py` (alongside `CONF_WEATHER_ENTITY`)
- [x] 1.3 Import and pass `CONF_PERSON_ENTITY` through `async_setup` into `hass.data[DOMAIN]`

## 2. Sensor: person entity subscription

- [x] 2.1 In `KleidungsempfehlungSensor.__init__`, store `config_data.get("person_entity")` as `self._person_entity`
- [x] 2.2 In `async_added_to_hass`, subscribe to `self._person_entity` via `async_track_state_change_event` when it is set (alongside existing entity subscriptions)

## 3. Sensor: parameter resolution

- [x] 3.1 Add helper `_get_person_entity_attr(attr_name, cast=float)` that reads `self._person_entity` state attributes, returns `None` on missing/unavailable/non-castable value, and logs a warning on coercion failure
- [x] 3.2 In `_async_update`, resolve `met_rate` using priority order: person entity `met_rate` attr → activity sensor → default `1.2`
- [x] 3.3 In `_async_update`, resolve `pmv_target` using priority order: person entity `pmv_target` attr → `person_config.get(CONF_PMV_TARGET, DEFAULT_PMV_TARGET)`
- [x] 3.4 In `_async_update`, resolve `age` using priority order: person entity `age` attr (cast to int) → `sensor_age` entity → `None`
- [x] 3.5 In `_async_update`, resolve `gender` using priority order: person entity `gender` attr (string) → `sensor_gender` entity → `None`

## 4. Config flow (UI)

- [x] 4.1 Add optional `person_entity` field to `config_flow.py` using an entity selector filtered to `person` domain

## 5. Documentation and example config

- [x] 5.1 Update `example_configuration.yaml` to show the `person_entity` key and a matching `homeassistant.customize` example with `met_rate`, `pmv_target`, `age`, and `gender` attributes
- [x] 5.2 Update `README.md` to document the person entity feature

## 6. Validation

- [x] 6.1 Restart HA test instance and confirm sensor loads without errors when `person_entity` is absent
- [x] 6.2 Add `person_entity: person.stub` to `ha-test/configuration.yaml`, add `homeassistant.customize` entry with `met_rate: 2.0`, `pmv_target: -0.5`, `age: 30`, and `gender: female`, restart, and confirm sensor attributes reflect those values
- [x] 6.3 Run `HA_TOKEN=$(cat .ha_token) ./ha-test/validate.sh` and confirm exit 0
