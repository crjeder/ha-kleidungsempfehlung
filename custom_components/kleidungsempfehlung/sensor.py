"""Sensor platform for Kleidungsempfehlung."""
from __future__ import annotations

import logging
import math
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN,
    CONF_WEATHER_ENTITY,
    CONF_SENSOR_TEMPERATURE,
    CONF_SENSOR_TEMPERATURE_HIGH,
    CONF_SENSOR_HUMIDITY,
    CONF_SENSOR_WIND,
    CONF_SENSOR_RAIN,
    CONF_SENSOR_RADIATION,
    CONF_SENSOR_ACTIVITY,
    CONF_SENSOR_AGE,
    CONF_SENSOR_GENDER,
    CONF_PMV_TARGET,
    DEFAULT_PMV_TARGET,
)
from .engine import (
    SmartClothingEngine,
    Weather,
    SLOTS,
    adjust_met_for_demographics,
    get_item_id,
    make_entry,
)

_LOGGER = logging.getLogger(__name__)


def _apparent_temperature(t_c: float, rh_pct: float = 50.0, ws_ms: float = 0.0) -> float:
    """Compute BOM apparent temperature (°C).

    AT = T + 0.33*e − 0.70*ws − 4.00
    e  = rh/100 * 6.105 * exp(17.27*T / (237.7+T))   [hPa]
    """
    e = (rh_pct / 100.0) * 6.105 * math.exp(17.27 * t_c / (237.7 + t_c))
    return t_c + 0.33 * e - 0.70 * ws_ms - 4.00


def _kmh_to_ms(kmh: float) -> float:
    """Convert km/h to m/s."""
    return kmh / 3.6


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the sensor platform from YAML configuration."""
    if DOMAIN not in hass.data:
        _LOGGER.error("Kleidungsempfehlung not configured in configuration.yaml")
        return

    data = hass.data[DOMAIN]
    entity = KleidungsempfehlungSensor(hass, data)
    async_add_entities([entity], True)


class KleidungsempfehlungSensor(RestoreEntity, SensorEntity):
    """Sensor that provides clothing recommendations based on weather."""

    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant, config_data: dict):
        """Initialize the sensor."""
        self.hass = hass
        self._config = config_data
        self._attr_name = config_data.get("name", "Kleidungsempfehlung")
        self._attr_unique_id = f"{DOMAIN}_recommendation"
        self._state = None
        self._attributes: dict[str, Any] = {}
        self._listeners = []

        # Initialize engine
        self._engine = SmartClothingEngine(
            inventory=config_data["inventory"],
            max_layers=config_data["max_layers"],
            solver=config_data["solver"],
            layering_factor=config_data["layering_factor"],
        )

        # Build base_ensemble from config
        self._base_ensemble = self._build_base_ensemble(config_data.get("base_ensemble", []))

    def _build_base_ensemble(self, base_config: list) -> dict | None:
        """Build base_ensemble dict from configuration."""
        if not base_config:
            return None

        ensemble = {slot: [None] * self._config["max_layers"] for slot in SLOTS}
        for entry in base_config:
            slot = entry["slot"]
            layer = entry["layer"]
            item_id = entry["item_id"]
            locked = entry.get("locked", False)
            if 0 <= layer < self._config["max_layers"]:
                ensemble[slot][layer] = make_entry(item_id, locked=locked)

        return ensemble

    async def async_added_to_hass(self) -> None:
        """Register callbacks for configured sensor entities."""
        await super().async_added_to_hass()

        # Collect all sensor entity IDs to monitor
        weather_sensors = self._config.get("weather_sensors", {})
        person_config = self._config.get("person", {})

        entity_ids = []
        for key in [CONF_SENSOR_TEMPERATURE, CONF_SENSOR_TEMPERATURE_HIGH,
                    CONF_SENSOR_HUMIDITY, CONF_SENSOR_WIND, CONF_SENSOR_RAIN,
                    CONF_SENSOR_RADIATION]:
            if entity_id := weather_sensors.get(key):
                entity_ids.append(entity_id)

        for key in [CONF_SENSOR_ACTIVITY, CONF_SENSOR_AGE, CONF_SENSOR_GENDER]:
            if entity_id := person_config.get(key):
                entity_ids.append(entity_id)

        if entity_ids:
            self._listeners.append(
                async_track_state_change_event(self.hass, entity_ids, self._async_inputs_updated)
            )

        # Subscribe to weather entity if configured
        if weather_entity_id := self._config.get("weather_entity"):
            self._listeners.append(
                async_track_state_change_event(
                    self.hass, [weather_entity_id], self._async_inputs_updated
                )
            )

        # Restore previous state
        if old_state := await self.async_get_last_state():
            try:
                self._state = float(old_state.state)
            except (ValueError, TypeError):
                self._state = old_state.state

        # Initial computation
        await self._async_update()

    async def async_will_remove_from_hass(self) -> None:
        """Unregister callbacks."""
        for unsub in self._listeners:
            try:
                unsub()
            except Exception:
                pass
        self._listeners = []

    @callback
    async def _async_inputs_updated(self, event) -> None:
        """Handle sensor state changes."""
        await self._async_update()

    def _get_sensor_value(self, entity_id: str | None, cast=float, default=None):
        """Get value from a sensor entity."""
        if not entity_id:
            return default
        state = self.hass.states.get(entity_id)
        if not state or state.state in ("unknown", "unavailable"):
            return default
        try:
            return cast(state.state)
        except (ValueError, TypeError):
            return state.state if cast == str else default

    def _get_met_rate(self) -> float:
        """Get metabolic rate from activity sensor or default."""
        person_config = self._config.get("person", {})
        activity_entity = person_config.get(CONF_SENSOR_ACTIVITY)

        if not activity_entity:
            return 1.2  # Default: light office work

        val = self._get_sensor_value(activity_entity, cast=str, default=None)
        if val is None:
            return 1.2

        # Try numeric value first
        try:
            return float(val)
        except (ValueError, TypeError):
            pass

        # Map common activity descriptions to Met values
        val_lower = str(val).lower()
        activity_map = {
            ("ruh", "sit", "sitz"): 1.0,      # Resting, sitting
            ("schlaf", "lieg"): 0.8,           # Sleeping
            ("steh", "leicht"): 1.2,           # Standing, light activity
            ("lang", "gemüt", "spazier"): 2.0, # Walking slowly
            ("zügig", "schnell", "wandern"): 3.0,  # Walking fast
            ("lauf", "jogg", "sport"): 4.0,    # Running, sports
        }

        for keywords, met in activity_map.items():
            if any(kw in val_lower for kw in keywords):
                return met

        return 1.2  # Default

    async def _async_fetch_forecast(self, entity_id: str) -> list:
        """Fetch hourly forecast from a weather entity.

        Uses the weather.get_forecasts service (HA ≥ 2023.9).
        Falls back to state.attributes['forecast'] on older HA.
        Returns an empty list on any error.
        """
        try:
            result = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": entity_id, "type": "hourly"},
                blocking=True,
                return_response=True,
            )
            forecasts = (result or {}).get(entity_id, {}).get("forecast", [])
            return forecasts or []
        except Exception as err:  # noqa: BLE001
            # ServiceNotFound on HA < 2023.9, or entity unavailable
            _LOGGER.warning(
                "weather.get_forecasts unavailable for %s (%s); falling back to state attribute",
                entity_id,
                err,
            )
            state = self.hass.states.get(entity_id)
            if state:
                return state.attributes.get("forecast", [])
            return []

    def _extract_weather_from_forecast(
        self, forecasts: list, wind_speed_unit: str = "km/h"
    ) -> tuple[float, float, float | None]:
        """Compute cold and warm temperatures from a forecast list.

        Returns (cold_t, cold_ws_ms, hot_t_or_None).
        cold_t  = apparent temperature of the coldest entry
        cold_ws = wind speed (m/s) of that same entry
        hot_t   = maximum raw temperature across all entries (None if only 1 entry)
        """
        if not forecasts:
            return (None, 0.0, None)

        best_apparent = float("inf")
        cold_t = None
        cold_ws = 0.0
        max_raw_t = float("-inf")

        for entry in forecasts:
            t = entry.get("temperature")
            if t is None:
                continue

            rh = entry.get("humidity", 50.0) or 50.0
            ws_raw = entry.get("wind_speed") or 0.0

            # Convert wind speed to m/s
            if wind_speed_unit == "km/h":
                ws_ms = _kmh_to_ms(ws_raw)
            else:
                ws_ms = float(ws_raw)

            apparent = _apparent_temperature(float(t), float(rh), ws_ms)

            if apparent < best_apparent:
                best_apparent = apparent
                cold_t = apparent
                cold_ws = ws_ms

            if float(t) > max_raw_t:
                max_raw_t = float(t)

        if cold_t is None:
            return (None, 0.0, None)

        hot_t = max_raw_t if len(forecasts) > 1 else None
        return (cold_t, cold_ws, hot_t)

    async def _async_update(self) -> None:
        """Compute clothing recommendation."""
        weather_entity_id = self._config.get("weather_entity")
        weather_sensors = self._config.get("weather_sensors", {})
        person_config = self._config.get("person", {})

        forecast_window_attrs: dict | None = None

        try:
            if weather_entity_id:
                # --- Weather-entity path ---
                forecasts = await self._async_fetch_forecast(weather_entity_id)

                # Determine wind speed unit from entity state attributes
                we_state = self.hass.states.get(weather_entity_id)
                wind_unit = "km/h"
                if we_state:
                    wind_unit = we_state.attributes.get("wind_speed_unit", "km/h")

                cold_t, cold_ws, hot_t = self._extract_weather_from_forecast(
                    forecasts, wind_speed_unit=wind_unit
                )

                if cold_t is None:
                    # Forecast empty — fall back to current state temperature
                    _LOGGER.warning(
                        "Forecast for %s is empty; using current state temperature",
                        weather_entity_id,
                    )
                    if we_state and we_state.state not in ("unknown", "unavailable"):
                        try:
                            cold_t = float(we_state.attributes.get("temperature", 20.0))
                        except (ValueError, TypeError):
                            cold_t = 20.0
                    else:
                        cold_t = 20.0
                    cold_ws = 0.0
                    hot_t = None
                else:
                    forecast_window_attrs = {
                        "entries": len(forecasts),
                        "t_min_perceived": round(cold_t, 2),
                        "t_max_raw": round(hot_t, 2) if hot_t is not None else cold_t,
                        "wind_at_min": round(cold_ws, 2),
                    }

                # Shared humidity/rain/radiation from individual sensors if configured
                humidity = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_HUMIDITY), float, 50.0
                )
                rain = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_RAIN), float, 0.0
                )
                radiation = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_RADIATION), float, None
                )

                t_ambient = cold_t
                wind_speed = cold_ws
                t_ambient_high = hot_t

                weather = Weather(
                    t_ambient=t_ambient,
                    wind_speed_ms=wind_speed,
                    rel_humidity=humidity,
                    rain_mm_h=rain,
                    t_radiant=radiation,
                )

                weather_high = None
                if t_ambient_high is not None:
                    weather_high = Weather(
                        t_ambient=t_ambient_high,
                        wind_speed_ms=0.0,
                        rel_humidity=humidity,
                        rain_mm_h=rain,
                        t_radiant=radiation,
                    )

            else:
                # --- Individual sensor path (original behaviour) ---
                t_ambient = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_TEMPERATURE), float, 20.0
                )
                t_ambient_high = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_TEMPERATURE_HIGH), float, None
                )
                humidity = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_HUMIDITY), float, 50.0
                )
                wind_speed = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_WIND), float, 0.0
                )
                rain = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_RAIN), float, 0.0
                )
                radiation = self._get_sensor_value(
                    weather_sensors.get(CONF_SENSOR_RADIATION), float, None
                )

                weather = Weather(
                    t_ambient=t_ambient,
                    wind_speed_ms=wind_speed,
                    rel_humidity=humidity,
                    rain_mm_h=rain,
                    t_radiant=radiation,
                )

                weather_high = None
                if t_ambient_high is not None:
                    weather_high = Weather(
                        t_ambient=t_ambient_high,
                        wind_speed_ms=wind_speed,
                        rel_humidity=humidity,
                        rain_mm_h=rain,
                        t_radiant=radiation,
                    )

            # Get metabolic rate and adjust for demographics
            base_met = self._get_met_rate()

            age = self._get_sensor_value(person_config.get(CONF_SENSOR_AGE), float, None)
            gender = self._get_sensor_value(person_config.get(CONF_SENSOR_GENDER), str, None)
            is_female = gender and any(x in gender.lower() for x in ["f", "w", "weib"])

            met_rate = adjust_met_for_demographics(
                base_met,
                age=int(age) if age else None,
                is_female=is_female
            )

            # Get PMV target
            pmv_target = person_config.get(CONF_PMV_TARGET, DEFAULT_PMV_TARGET)

            # Get recommendation
            result = self._engine.recommend_outfit(
                weather=weather,
                weather_high=weather_high,
                met_rate=met_rate,
                pmv_target=pmv_target,
                base_ensemble=self._base_ensemble,
            )

            # Format ensemble for attributes
            ensemble_formatted = {}
            for slot in SLOTS:
                items = []
                for idx, entry in enumerate(result.ensemble.get(slot, [])):
                    item_id = get_item_id(entry)
                    if item_id:
                        items.append({"layer": idx, "id": item_id})
                if items:
                    ensemble_formatted[slot] = items

            self._state = round(result.achieved_clo, 2)
            self._attributes = {
                "ensemble": ensemble_formatted,
                "target_clo": result.target_clo,
                "target_clo_warm": result.target_clo_warm,
                "achieved_clo": result.achieved_clo,
                "pmv": result.pmv,
                "ppd": result.ppd,
                "weather": {
                    "temperature": t_ambient,
                    "temperature_high": t_ambient_high,
                    "humidity": humidity,
                    "wind": wind_speed,
                    "rain": rain,
                },
                "met_rate": met_rate,
                "pmv_target": pmv_target,
            }
            if forecast_window_attrs is not None:
                self._attributes["forecast_window"] = forecast_window_attrs

        except Exception as err:
            _LOGGER.exception("Error computing clothing recommendation: %s", err)
            self._state = None
            self._attributes = {"error": str(err)}

        self.async_write_ha_state()

    @property
    def native_value(self):
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        return self._attributes

    @property
    def icon(self) -> str:
        """Return the icon."""
        return "mdi:tshirt-crew"

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return "clo"
