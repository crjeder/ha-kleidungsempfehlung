## Context

Currently `met_rate` (via activity sensor), `pmv_target`, `age` (via age sensor), and `gender` (via gender sensor) are configured under the `person:` key in `configuration.yaml`. This means each HA instance must repeat these values in the sensor config, with no connection to the HA `person` entity representing the actual person.

HA supports custom attributes on any entity via `homeassistant.customize` in `configuration.yaml`:
```yaml
homeassistant:
  customize:
    person.me:
      met_rate: 1.4
      pmv_target: -0.3
```
These appear as extra attributes on the person entity's state object (`hass.states.get("person.me").attributes`).

The integration currently has a `person:` config section for sensor-based activity/age/gender lookups but no direct link to a `person.*` entity. This change adds an optional `person_entity` config key that, when set, causes the sensor to read `met_rate`, `pmv_target`, `age`, and `gender` directly from that entity's custom attributes.

## Goals / Non-Goals

**Goals:**
- Read `met_rate`, `pmv_target`, `age`, and `gender` from a `person.*` entity's custom attributes when `person_entity` is configured.
- Fall back to sensor config values when person entity attributes are absent or unavailable.
- Track state changes on the person entity so the sensor recalculates when attributes update (note: HA only fires `state_changed` when state or attributes change).
- Add `person_entity` to the voluptuous config schema and the UI config flow.

**Non-Goals:**
- Writing back to the person entity.
- Supporting multiple person entities per sensor.
- Changing engine.py or any CLI behavior.
- Supporting multiple person entities per sensor.
- Writing back to the person entity.

## Decisions

### 1. Attribute names on the person entity

Use `met_rate`, `pmv_target`, `age`, and `gender` as the attribute names — identical to the existing config keys. This avoids a translation layer and is self-documenting.

`gender` is a string (e.g. `"female"`, `"male"`, `"w"`, `"f"`) matched with the same keyword logic already present in `sensor.py`.

**Alternative**: Use a nested namespace like `kleidungsempfehlung_met_rate`. Rejected — overly verbose, and custom attributes on `person` are user-managed so naming collisions are the user's responsibility.

### 2. Priority: person entity attributes override config values

When `person_entity` is set and the entity has a `met_rate` (or `age`, `gender`, `pmv_target`) attribute, that value wins over the corresponding `person:` config section entry. Config section acts as fallback.

**Alternative**: Merge (average). Rejected — not meaningful for these parameters.

### 3. State-change tracking

Subscribe to the `person_entity` via `async_track_state_change_event` alongside other entities. HA fires `state_changed` when the entity's attributes change (e.g., via a service call), so this is sufficient.

**Caveat**: `homeassistant.customize` attributes are set at startup and don't change dynamically — they won't trigger `state_changed`. The subscription is still correct for completeness and future-proofing (e.g., a template or automation updating a person attribute).

### 4. Schema changes

- `person_entity` added as `vol.Optional` at the top level of `CONFIG_SCHEMA` (alongside `weather_entity`).
- `met_rate` added as `vol.Optional` to `PERSON_SCHEMA` (direct numeric override, bypassing the activity sensor).

### 5. Resolution order for `met_rate`

1. Person entity attribute `met_rate` (if `person_entity` configured and attribute present)
2. Activity sensor (`sensor_activity` in `person:` config) — existing behavior
3. Default: `1.2`

Resolution order for `pmv_target`:
1. Person entity attribute `pmv_target` (if `person_entity` configured and attribute present)
2. `person.pmv_target` from config
3. Default: `0.0`

Resolution order for `age`:
1. Person entity attribute `age` (if `person_entity` configured and attribute present)
2. Age sensor (`sensor_age` in `person:` config) — existing behavior
3. Default: `None` (no demographic adjustment)

Resolution order for `gender`:
1. Person entity attribute `gender` (if `person_entity` configured and attribute present)
2. Gender sensor (`sensor_gender` in `person:` config) — existing behavior
3. Default: `None` (no demographic adjustment)

## Risks / Trade-offs

- **Attribute not present at startup**: Person entity may not have `met_rate`/`pmv_target` attributes if the user hasn't added `customize` entries yet. Fallback to config/default mitigates this gracefully.
- **Type coercion**: Custom attributes from `customize` are parsed as YAML scalars so they arrive as Python `float`/`int`/`str`. The sensor must coerce to `float` with error handling to avoid crashing on a bad value.
- **`homeassistant.customize` is static**: Values set via `customize` do not fire `state_changed` because they are baked in at startup as part of the entity's initial attributes. This is acceptable — the recommendation will still be correct after the next state change from another tracked entity (weather, activity, etc.) or HA restart.

## Migration Plan

No migration needed. The `person_entity` key is optional. Existing configs continue to work without change.
