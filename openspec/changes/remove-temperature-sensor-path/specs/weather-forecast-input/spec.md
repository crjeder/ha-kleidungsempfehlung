## MODIFIED Requirements

### Requirement: Weather entity configuration
The integration SHALL require a `weather_entity` configuration key specifying a HA `weather.*` entity ID. It is the sole source for temperature and wind data. Configuring the integration without `weather_entity` SHALL produce a validation error at startup.

#### Scenario: Valid weather entity configured
- **WHEN** `weather_entity: weather.home` is set in configuration
- **THEN** the integration loads without error and the sensor becomes available

#### Scenario: No weather entity configured
- **WHEN** `weather_entity` is absent from configuration
- **THEN** the integration raises a configuration validation error and does not start

#### Scenario: Invalid entity ID configured
- **WHEN** `weather_entity` is set to a non-existent entity ID
- **THEN** the sensor logs a warning and uses the forecast fallback path (current state temperature); it does NOT fall back to individual sensor entities

## REMOVED Requirements

### Requirement: Individual sensor temperature path
**Reason**: The `weather_entity` forecast path supersedes it. Maintaining two input paths adds unnecessary complexity and divergent behaviour.
**Migration**: Add `weather_entity: weather.<your_provider>` to configuration.yaml and remove `weather_sensors.sensor_temperature`, `sensor_temperature_high`, `sensor_wind`, `sensor_rain`, and `sensor_radiation` keys. `sensor_humidity` may be retained as an optional local override.
