### Requirement: Person entity configuration
The integration SHALL accept an optional `person_entity` key (a valid HA entity ID of the form `person.*`) in the top-level YAML config. When present, the sensor SHALL attempt to read clothing recommendation parameters from that entity's state attributes.

#### Scenario: Valid person entity in config
- **WHEN** `configuration.yaml` includes `person_entity: person.alice`
- **THEN** the integration loads without error and stores the entity ID for use at runtime

#### Scenario: Missing person entity key
- **WHEN** `configuration.yaml` does not include `person_entity`
- **THEN** the integration behaves identically to the current behavior with no person entity lookup

### Requirement: met_rate from person entity attributes
When `person_entity` is configured, the sensor SHALL read the `met_rate` attribute from the person entity and use it as the base metabolic rate, taking precedence over the activity sensor and the `person.met_rate` static config value.

#### Scenario: met_rate attribute present on person entity
- **WHEN** `person_entity` is configured and `hass.states.get(person_entity).attributes` contains `met_rate: 1.6`
- **THEN** the sensor uses `1.6` as the base Met value (before demographic adjustment)

#### Scenario: met_rate attribute absent from person entity
- **WHEN** `person_entity` is configured but the person entity has no `met_rate` attribute
- **THEN** the sensor falls back to the activity sensor value, or the default of `1.2` if no activity sensor is configured

#### Scenario: met_rate attribute is non-numeric
- **WHEN** `person_entity` is configured and the person entity's `met_rate` attribute is a non-numeric string (e.g. `"active"`)
- **THEN** the sensor logs a warning and falls back to the activity sensor / default without crashing

### Requirement: pmv_target from person entity attributes
When `person_entity` is configured, the sensor SHALL read the `pmv_target` attribute from the person entity and use it as the PMV comfort target, taking precedence over the `person.pmv_target` static config value.

#### Scenario: pmv_target attribute present on person entity
- **WHEN** `person_entity` is configured and `hass.states.get(person_entity).attributes` contains `pmv_target: -0.3`
- **THEN** the sensor uses `-0.3` as the PMV target for the recommendation

#### Scenario: pmv_target attribute absent from person entity
- **WHEN** `person_entity` is configured but the person entity has no `pmv_target` attribute
- **THEN** the sensor falls back to `person.pmv_target` from config, or the default of `0.0`

#### Scenario: pmv_target attribute is non-numeric
- **WHEN** `person_entity` is configured and the person entity's `pmv_target` attribute is a non-numeric value
- **THEN** the sensor logs a warning and falls back to config / default without crashing

### Requirement: Person entity state change triggers recalculation
When `person_entity` is configured, the sensor SHALL subscribe to state changes on the person entity and recalculate the clothing recommendation whenever the entity's state or attributes change.

#### Scenario: Person entity state changes
- **WHEN** the person entity fires a `state_changed` event (e.g. presence changes or an automation updates its attributes)
- **THEN** the sensor recalculates the clothing recommendation using the latest person entity attributes

### Requirement: age from person entity attributes
When `person_entity` is configured, the sensor SHALL read the `age` attribute from the person entity and use it for demographic Met adjustment, taking precedence over the `sensor_age` entity in the `person:` config section.

#### Scenario: age attribute present on person entity
- **WHEN** `person_entity` is configured and the person entity's attributes contain `age: 35`
- **THEN** the sensor uses `35` as the age for `adjust_met_for_demographics`

#### Scenario: age attribute absent from person entity
- **WHEN** `person_entity` is configured but the person entity has no `age` attribute
- **THEN** the sensor falls back to the `sensor_age` entity value, or `None` (no adjustment) if neither is configured

#### Scenario: age attribute is non-numeric
- **WHEN** `person_entity` is configured and the person entity's `age` attribute cannot be cast to int
- **THEN** the sensor logs a warning and treats age as absent (no demographic adjustment)

### Requirement: gender from person entity attributes
When `person_entity` is configured, the sensor SHALL read the `gender` attribute from the person entity and use it for demographic Met adjustment, taking precedence over the `sensor_gender` entity in the `person:` config section.

#### Scenario: gender attribute present on person entity
- **WHEN** `person_entity` is configured and the person entity's attributes contain `gender: female`
- **THEN** the sensor interprets this using the existing female-keyword matching logic (`f`, `w`, `weib`) and applies the demographic adjustment

#### Scenario: gender attribute absent from person entity
- **WHEN** `person_entity` is configured but the person entity has no `gender` attribute
- **THEN** the sensor falls back to the `sensor_gender` entity value, or `None` (no adjustment) if neither is configured

### Requirement: Resolved parameters exposed in sensor attributes
The sensor's `extra_state_attributes` SHALL include the `met_rate` and `pmv_target` values that were actually used for the recommendation, regardless of whether they originated from the person entity or from config.

#### Scenario: Attribute source transparency
- **WHEN** `met_rate` was sourced from the person entity attribute
- **THEN** `sensor.kleidungsempfehlung.attributes.met_rate` reflects that value (same as current behavior — value is already exposed)
