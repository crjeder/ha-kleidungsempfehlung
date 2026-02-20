from __future__ import annotations

from typing import Any, Dict, Optional

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.selector import selector

from .const import DOMAIN, DEFAULT_NAME

# Keys for options
INPUT_KEYS = [
    "sensor_uv",
    "sensor_praeferenz",
    "sensor_geschlecht",
    "sensor_alter",
    "sensor_gewicht",
    "sensor_groesse",
    "sensor_wind",
    "sensor_temperatur",
    "sensor_luftfeuchte",
    "sensor_sonnenstrahlung",
    "sensor_aktivitaet"
]


class KleidungsempfehlungConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1
    def __init__(self):
        self._values: Dict[str, Any] = {}

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None):
        """Handle the initial step."""
        if user_input is not None:
            # Create the entry; store configured entity IDs in options so OptionsFlow can edit later
            return self.async_create_entry(title=DEFAULT_NAME, data={}, options=user_input)

        data_schema = vol.Schema(
            {
                vol.Optional("sensor_temperatur"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_luftfeuchte"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_wind"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_uv"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_sonnenstrahlung"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_aktivitaet"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_praeferenz"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_geschlecht"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_alter"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_gewicht"): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_groesse"): selector({"entity": {"domain": "sensor"}}),
            }
        )
        return self.async_show_form(step_id="user", data_schema=data_schema)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self.config_entry = config_entry
        # use existing options as defaults
        self._options = dict(config_entry.options or {})

    async def async_step_init(self, user_input: Optional[Dict[str, Any]] = None):
        if user_input is not None:
            self._options.update(user_input)
            return self.async_create_entry(title="", data=self._options)

        data_schema = vol.Schema(
            {
                vol.Optional("sensor_temperatur", default=self._options.get("sensor_temperatur")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_luftfeuchte", default=self._options.get("sensor_luftfeuchte")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_wind", default=self._options.get("sensor_wind")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_uv", default=self._options.get("sensor_uv")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_sonnenstrahlung", default=self._options.get("sensor_sonnenstrahlung")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_aktivitaet", default=self._options.get("sensor_aktivitaet")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_praeferenz", default=self._options.get("sensor_praeferenz")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_geschlecht", default=self._options.get("sensor_geschlecht")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_alter", default=self._options.get("sensor_alter")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_gewicht", default=self._options.get("sensor_gewicht")): selector({"entity": {"domain": "sensor"}}),
                vol.Optional("sensor_groesse", default=self._options.get("sensor_groesse")): selector({"entity": {"domain": "sensor"}}),
            }
        )
        return self.async_show_form(step_id="init", data_schema=data_schema)