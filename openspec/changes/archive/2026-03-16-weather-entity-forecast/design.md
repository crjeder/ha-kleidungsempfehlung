## Context

The integration currently reads weather from individual HA sensor entities (temperature, wind, humidity, rain, radiation). Users who already have a HA weather integration (e.g. `weather.home`, Met.no, OpenWeatherMap) would prefer to point at a single `weather` entity and have the engine use the **worst expected conditions** across the forecast window rather than the instantaneous reading.

HA's `WeatherEntity` exposes `async_forecast_hourly()` which returns a list of `Forecast` dicts with at minimum `datetime`, `temperature`, and optionally `wind_speed`, `precipitation`, `humidity`. The perceived temperature (apparent temperature / wind-chill) can be approximated from air temperature and wind speed.

Current state: `sensor.py` reads individual sensor entity IDs from `weather_sensors` config block. The engine `recommend_outfit()` receives a `Weather` object for the cold scenario and optionally a `Weather` object for the warm scenario (range mode).

## Goals / Non-Goals

**Goals:**
- Accept a `weather_entity` config key pointing to a HA `weather.*` entity
- Fetch hourly forecast via `async_forecast_hourly` on each update cycle
- Compute the perceived (apparent) temperature for each forecast entry using a standard wind-chill / apparent-temperature formula
- Use the **minimum perceived temperature** across the forecast window as `t_ambient` (cold scenario)
- Use the **maximum temperature** across the forecast window as `t_ambient_high` (warm/range scenario), if applicable
- Forward the corresponding wind speed alongside the chosen temperatures into the `Weather` objects
- Keep existing `weather_sensors` config path fully functional as fallback when no `weather_entity` is set

**Non-Goals:**
- Precipitation probability / radar integration
- Caching forecast data between update cycles
- Configurable forecast look-ahead window (always use all returned hours, typically 24–48h)
- Replacing the existing individual sensor path with a mandatory weather entity

## Decisions

### D1: Perceived temperature formula
Use the standard **Australian BOM apparent temperature** formula for combined wind-chill and humidity effect:

```
AT = T + 0.33 * e - 0.70 * ws - 4.00
where e = rh/100 * 6.105 * exp(17.27*T/(237.7+T))   (vapour pressure, hPa)
      ws = wind speed in m/s
```

This is preferred over the simpler NOAA wind-chill (valid only below 10 °C) because the engine already uses PMV which is relevant across all seasons.

Fallback: if wind speed is missing from a forecast entry, treat it as 0 m/s (no wind chill).

### D2: Forecast entity subscription
Use `async_forecast_hourly` via `hass.components.weather.async_get_forecast` helper. HA ≥ 2023.9 introduced `async_forecast_hourly` as a method on `WeatherEntity` that integrations must call through the service layer (`weather.get_forecasts` service or entity method). We will call the entity method directly via `hass.states` + entity registry lookup, then subscribe to state changes of the weather entity to trigger re-computation.

Implementation detail: obtain the entity object via `hass.data[DOMAIN]["weather_entity_id"]` → look up with `er.async_get(entity_id)` → call `platform_entity.async_forecast_hourly()`. If the entity is not available, fall back to the state's `forecast` attribute (some integrations still populate it).

### D3: Config precedence
When both `weather_entity` and `weather_sensors` temperature fields are configured, `weather_entity` takes precedence for temperature and wind. Other sensor fields (humidity, rain, radiation) still read from individual sensors if configured — the weather entity forecast may lack humidity or precipitation intensity.

### D4: Warm scenario (range mode)
When a `weather_entity` is configured, automatically enable range mode using:
- `t_ambient` = min perceived temperature across forecast
- `t_ambient_high` = max raw temperature across forecast (perceived warmth at rest is less relevant for the warm scenario)

This mirrors how manual `sensor_temperature_high` is used today.

### D5: Accessing the WeatherEntity object
HA does not expose `WeatherEntity` instances directly from `hass.states`. The correct approach for calling `async_forecast_hourly` is via the `weather.get_forecasts` **service call** (added in HA 2023.9) or by accessing the entity through the entity platform. We will use the service call approach as it works across all weather integrations without coupling to internal platform state:

```python
result = await hass.services.async_call(
    "weather", "get_forecasts",
    {"entity_id": entity_id, "type": "hourly"},
    blocking=True, return_response=True
)
forecasts = result[entity_id]["forecast"]
```

## Risks / Trade-offs

- **HA version dependency** → `weather.get_forecasts` service exists since HA 2023.9. If called on older HA, the service call raises `ServiceNotFound`. Mitigation: catch `ServiceNotFound` and fall back to `state.attributes.get("forecast", [])` with a deprecation log.
- **Forecast unavailable** → Some weather integrations may return an empty forecast list (e.g. during startup). Mitigation: if forecast is empty, fall back to the current state temperature of the weather entity.
- **Wind speed units** → HA weather entities report wind in km/h by default; the engine expects m/s. Conversion: `ws_ms = ws_kmh / 3.6`. Check `state.attributes["wind_speed_unit"]` if present.
- **Blocking async call** → `async_call(blocking=True)` runs in the event loop but waits for the service to complete. This is acceptable for a sensor update; do not call from `@callback` context.

## Migration Plan

1. Deploy new code — existing YAML configs with only `weather_sensors` continue to work unchanged.
2. Users add `weather_entity: weather.home` to their config and optionally remove redundant `sensor_temperature` / `sensor_wind` entries.
3. No migration of stored state needed; sensor attributes gain new `forecast_window` field but existing consumers ignore unknown attributes.

## Open Questions

- Should the forecast look-ahead window be configurable (e.g., next 6h vs next 24h)? Deferred — use all returned hours for now.
- Should perceived temperature or raw temperature be used for the warm scenario? Currently decided: raw max temperature for warm, perceived min for cold. Re-evaluate after user feedback.
