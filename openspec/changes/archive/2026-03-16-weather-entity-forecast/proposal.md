## Why

The HA integration currently accepts a temperature sensor entity, but real-world clothing decisions should account for the coldest part of the upcoming day rather than a single point-in-time reading. By consuming a HA weather entity with hourly forecast data, the engine can compute the minimum *perceived* (apparent/wind-chill) temperature across the forecast window and dress for the worst expected conditions.

## What Changes

- Add a `weather_entity` config option (replaces or supplements bare `temperature_entity`)
- Call `async_forecast_hourly` on the weather entity to retrieve the hourly forecast
- Compute the minimum perceived temperature from the forecast (accounting for wind speed when available)
- Feed the computed temperature (and wind speed) into `engine.recommend_outfit()` instead of the live sensor reading
- Keep the existing `temperature_entity` path working as a fallback when no weather entity is configured

## Capabilities

### New Capabilities
- `weather-forecast-input`: Fetch hourly forecast from a HA weather entity, compute min perceived temperature, and use it as the thermal input to the recommendation engine

### Modified Capabilities
- (none — the engine interface and sensor output schema are unchanged)

## Impact

- **`custom_components/kleidungsempfehlung/sensor.py`**: Main change site — new forecast-fetch logic and perceived-temperature computation
- **`custom_components/kleidungsempfehlung/__init__.py`**: Config validation must accept `weather_entity` key
- **`custom_components/kleidungsempfehlung/const.py`**: New `CONF_WEATHER_ENTITY` constant
- **`custom_components/kleidungsempfehlung/config_flow.py`**: UI flow must expose the new field
- **Dependencies**: No new Python packages; relies on HA's built-in `weather` platform API (`WeatherEntity.async_forecast_hourly`)
- **HA minimum version**: Requires HA ≥ 2023.9 for `async_forecast_hourly` subscription API
