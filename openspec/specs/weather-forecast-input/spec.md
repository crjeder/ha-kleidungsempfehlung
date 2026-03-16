### Requirement: Weather entity configuration
The integration SHALL accept an optional `weather_entity` configuration key specifying a HA `weather.*` entity ID. When present, it SHALL be used as the source for temperature and wind data instead of individual sensor entities for those fields.

#### Scenario: Valid weather entity configured
- **WHEN** `weather_entity: weather.home` is set in configuration
- **THEN** the integration loads without error and the sensor becomes available

#### Scenario: No weather entity configured
- **WHEN** `weather_entity` is absent from configuration
- **THEN** the integration behaves identically to the current behaviour using `weather_sensors` entries

#### Scenario: Invalid entity ID configured
- **WHEN** `weather_entity` is set to a non-existent entity ID
- **THEN** the sensor logs a warning and falls back to the current `weather_sensors` path (no crash)

---

### Requirement: Hourly forecast fetch
When a `weather_entity` is configured, the sensor SHALL call `weather.get_forecasts` (with `type: hourly`) on each update cycle to retrieve the hourly forecast list.

#### Scenario: Forecast available
- **WHEN** the weather entity returns a non-empty hourly forecast
- **THEN** the sensor uses that forecast list for perceived-temperature computation

#### Scenario: Forecast service unavailable (HA < 2023.9)
- **WHEN** `weather.get_forecasts` raises `ServiceNotFound`
- **THEN** the sensor falls back to `state.attributes["forecast"]` and logs a deprecation warning

#### Scenario: Forecast list is empty
- **WHEN** the forecast list is empty (integration not ready yet)
- **THEN** the sensor falls back to the weather entity's current state temperature and wind speed and logs a warning

---

### Requirement: Perceived temperature computation
For each hourly forecast entry the sensor SHALL compute the apparent temperature using the BOM formula:

```
e  = rh/100 * 6.105 * exp(17.27 * T / (237.7 + T))   [hPa]
AT = T + 0.33 * e − 0.70 * ws − 4.00
```

where `T` is air temperature in °C, `rh` is relative humidity (%), and `ws` is wind speed in m/s. Missing humidity SHALL default to 50 %. Missing wind speed SHALL default to 0 m/s.

#### Scenario: Full data available
- **WHEN** a forecast entry contains temperature, humidity, and wind speed
- **THEN** the apparent temperature is computed with all three inputs

#### Scenario: Wind speed missing from entry
- **WHEN** a forecast entry has temperature and humidity but no wind speed
- **THEN** wind speed defaults to 0 m/s and apparent temperature is computed accordingly

#### Scenario: Wind speed in km/h
- **WHEN** the weather entity reports wind speed in km/h (the HA default)
- **THEN** the sensor converts to m/s (divide by 3.6) before applying the formula

---

### Requirement: Minimum perceived temperature selection
The sensor SHALL select the entry with the minimum apparent temperature across all forecast entries as the cold-scenario input to the engine.

#### Scenario: Cold scenario temperature
- **WHEN** forecast entries are processed
- **THEN** `t_ambient` passed to `recommend_outfit()` equals the minimum apparent temperature across all entries

#### Scenario: Cold scenario wind speed
- **WHEN** the minimum-apparent-temperature entry is selected
- **THEN** the wind speed from that same entry is passed as `wind_speed_ms` in the cold `Weather` object

---

### Requirement: Warm scenario from forecast
When a `weather_entity` is configured, the sensor SHALL automatically populate `weather_high` using the entry with the maximum raw (non-apparent) temperature across the forecast, enabling range mode without requiring explicit `sensor_temperature_high` configuration.

#### Scenario: Range mode auto-enabled
- **WHEN** a weather entity is configured and the forecast contains more than one entry
- **THEN** `weather_high` is passed to `recommend_outfit()` with the max raw temperature

#### Scenario: Single-entry forecast
- **WHEN** the forecast contains exactly one entry
- **THEN** `weather_high` is `None` (single-temperature mode)

---

### Requirement: Sensor attributes expose forecast metadata
The sensor's `extra_state_attributes` SHALL include a `forecast_window` key when a weather entity is used, summarising the data that drove the computation.

#### Scenario: Forecast metadata in attributes
- **WHEN** the sensor updates successfully from a weather entity forecast
- **THEN** attributes contain `forecast_window` with keys: `entries` (count), `t_min_perceived` (float), `t_max_raw` (float), `wind_at_min` (float)

---

### Requirement: State-change subscription for weather entity
When a `weather_entity` is configured, the sensor SHALL subscribe to state-change events for that entity and trigger a re-computation on each change, matching the existing behaviour for individual sensor entities.

#### Scenario: Weather entity state changes
- **WHEN** the configured weather entity fires a state-change event
- **THEN** `_async_update` is called and the sensor state is refreshed
