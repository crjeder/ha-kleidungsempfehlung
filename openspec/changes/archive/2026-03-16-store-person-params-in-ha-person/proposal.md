## Why

Currently, person-specific parameters (met_rate, pmv_target, age, gender) must be configured per-sensor in `configuration.yaml` — either as static values or as sensor entity IDs. This means the same person's data must be duplicated if they have multiple sensors or if the integration is reconfigured. Storing these on the HA `person` entity via `homeassistant.customize` lets the integration read them directly from the person object, giving a single source of truth per person.

## What Changes

- The sensor reads `met_rate`, `pmv_target`, `age`, and `gender` from the HA `person` entity's custom attributes when `person_entity` is configured.
- Person entity attributes take precedence over the corresponding `person:` config section values and sensor entity lookups.
- New optional config key `person_entity` added to the sensor/integration config.

## Capabilities

### New Capabilities

- `person-params`: Read clothing recommendation parameters (met_rate, pmv_target, age, gender) from HA person entity custom attributes, falling back to sensor config values.

### Modified Capabilities

<!-- No existing spec-level requirements are changing. -->

## Impact

- `custom_components/kleidungsempfehlung/sensor.py`: `_get_met_rate()`, PMV target resolution, age/gender lookups must check `person` entity state attributes when `person_entity` is configured.
- `custom_components/kleidungsempfehlung/const.py`: Add `CONF_PERSON_ENTITY` constant.
- `custom_components/kleidungsempfehlung/config_flow.py`: Add optional person entity selector.
- `custom_components/kleidungsempfehlung/__init__.py`: Allow `met_rate` / `pmv_target` to be omitted from sensor config when `person_entity` is set.
- `custom_components/kleidungsempfehlung/example_configuration.yaml`: Add example showing person entity usage.
- No changes to `engine.py` or standalone CLI.
