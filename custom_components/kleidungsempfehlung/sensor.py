"""Sensor platform for Kleidungsempfehlung."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN,
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

    async def _async_update(self) -> None:
        """Compute clothing recommendation."""
        weather_sensors = self._config.get("weather_sensors", {})
        person_config = self._config.get("person", {})

        try:
            # Read weather values
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

            # Build Weather object
            weather = Weather(
                t_ambient=t_ambient,
                wind_speed_ms=wind_speed,
                rel_humidity=humidity,
                rain_mm_h=rain,
                t_radiant=radiation,  # Use solar radiation as radiant temp if available
            )

            # Build weather_high if configured
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
