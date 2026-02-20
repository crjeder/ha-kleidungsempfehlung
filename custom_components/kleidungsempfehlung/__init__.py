"""Home Assistant integration for clothing recommendations based on weather."""
from __future__ import annotations

import logging
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.const import CONF_NAME

from .const import (
    DOMAIN,
    PLATFORMS,
    DEFAULT_NAME,
    CONF_INVENTORY,
    CONF_BASE_ENSEMBLE,
    CONF_MAX_LAYERS,
    CONF_SOLVER,
    CONF_LAYERING_FACTOR,
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
    DEFAULT_MAX_LAYERS,
    DEFAULT_SOLVER,
    DEFAULT_LAYERING_FACTOR,
    DEFAULT_PMV_TARGET,
)

_LOGGER = logging.getLogger(__name__)

# Schema for inventory items
INVENTORY_ITEM_SCHEMA = vol.Schema({
    vol.Required("id"): cv.string,
    vol.Required("name"): cv.string,
    vol.Required("slot"): vol.In(["head", "neck", "torso", "hands", "legs", "feet"]),
    vol.Required("fit"): vol.In(["base", "mid", "outer"]),
    vol.Required("clo"): vol.Coerce(float),
    vol.Optional("waterproof", default=False): cv.boolean,
    vol.Optional("windproof", default=False): cv.boolean,
})

# Schema for base_ensemble entries
BASE_ENSEMBLE_ENTRY_SCHEMA = vol.Schema({
    vol.Required("slot"): vol.In(["head", "neck", "torso", "hands", "legs", "feet"]),
    vol.Required("layer"): vol.Coerce(int),
    vol.Required("item_id"): cv.string,
    vol.Optional("locked", default=False): cv.boolean,
})

# Schema for weather sensors
WEATHER_SENSORS_SCHEMA = vol.Schema({
    vol.Required(CONF_SENSOR_TEMPERATURE): cv.entity_id,
    vol.Optional(CONF_SENSOR_TEMPERATURE_HIGH): cv.entity_id,
    vol.Optional(CONF_SENSOR_HUMIDITY): cv.entity_id,
    vol.Optional(CONF_SENSOR_WIND): cv.entity_id,
    vol.Optional(CONF_SENSOR_RAIN): cv.entity_id,
    vol.Optional(CONF_SENSOR_RADIATION): cv.entity_id,
})

# Schema for person sensors
PERSON_SCHEMA = vol.Schema({
    vol.Optional(CONF_SENSOR_ACTIVITY): cv.entity_id,
    vol.Optional(CONF_SENSOR_AGE): cv.entity_id,
    vol.Optional(CONF_SENSOR_GENDER): cv.entity_id,
    vol.Optional(CONF_PMV_TARGET, default=DEFAULT_PMV_TARGET): vol.Coerce(float),
})

# Main configuration schema
CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Required(CONF_INVENTORY): vol.All(cv.ensure_list, [INVENTORY_ITEM_SCHEMA]),
        vol.Optional(CONF_BASE_ENSEMBLE): vol.All(cv.ensure_list, [BASE_ENSEMBLE_ENTRY_SCHEMA]),
        vol.Optional(CONF_MAX_LAYERS, default=DEFAULT_MAX_LAYERS): vol.Coerce(int),
        vol.Optional(CONF_SOLVER, default=DEFAULT_SOLVER): vol.In(["ilp", "heuristic"]),
        vol.Optional(CONF_LAYERING_FACTOR, default=DEFAULT_LAYERING_FACTOR): vol.Coerce(float),
        vol.Required("weather"): WEATHER_SENSORS_SCHEMA,
        vol.Optional("person"): PERSON_SCHEMA,
    })
}, extra=vol.ALLOW_EXTRA)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Kleidungsempfehlung component from YAML configuration."""
    if DOMAIN not in config:
        return True

    conf = config[DOMAIN]
    hass.data[DOMAIN] = {
        "config": conf,
        "inventory": conf[CONF_INVENTORY],
        "base_ensemble": conf.get(CONF_BASE_ENSEMBLE, []),
        "max_layers": conf.get(CONF_MAX_LAYERS, DEFAULT_MAX_LAYERS),
        "solver": conf.get(CONF_SOLVER, DEFAULT_SOLVER),
        "layering_factor": conf.get(CONF_LAYERING_FACTOR, DEFAULT_LAYERING_FACTOR),
        "weather_sensors": conf["weather"],
        "person": conf.get("person", {}),
        "name": conf.get(CONF_NAME, DEFAULT_NAME),
    }

    # Load sensor platform
    hass.async_create_task(
        hass.helpers.discovery.async_load_platform("sensor", DOMAIN, {}, config)
    )

    _LOGGER.info("Kleidungsempfehlung integration loaded with %d inventory items", len(conf[CONF_INVENTORY]))
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up from a config entry (UI-based, for backwards compatibility)."""
    hass.data.setdefault(DOMAIN, {})
    for platform in PLATFORMS:
        hass.async_create_task(
            hass.config_entries.async_forward_entry_setup(entry, platform)
        )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = all(
        await hass.config_entries.async_forward_entry_unload(entry, platform)
        for platform in PLATFORMS
    )
    return unload_ok
