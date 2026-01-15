from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, DEFAULT_NAME
from .engine import recommend_clo_with_overlays
from homeassistant.helpers.restore_state import RestoreEntity

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=30)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback):
    options = dict(entry.options or {})
    entity = KleidungsempfehlungSensor(hass, entry, options)
    async_add_entities([entity], True)

class KleidungsempfehlungSensor(RestoreEntity, SensorEntity):
    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, options: dict):
        self.hass = hass
        self._entry = entry
        self._options = options
        self._attr_name = DEFAULT_NAME
        self._state = None
        self._attributes: dict[str, Any] = {}
        self._listeners = []

    async def async_added_to_hass(self):
        """Register callbacks for configured entities."""
        await super().async_added_to_hass()
        # listen to all configured entity IDs for changes
        entity_ids = [v for v in self._options.values() if v]
        if entity_ids:
            self._listeners.append(
                async_track_state_change_event(self.hass, entity_ids, self._async_inputs_updated)
            )
        # restore state if available
        old = await self.async_get_last_state()
        if old:
            try:
                self._state = float(old.state)
            except Exception:
                self._state = old.state

        # initial computation
        await self._async_update()

    async def async_will_remove_from_hass(self):
        for unsub in self._listeners:
            try:
                unsub()
            except Exception:
                pass
        self._listeners = []

    async def _async_inputs_updated(self, event):
        await self._async_update()

    async def _async_update(self):
        """Read inputs and compute recommendation."""
        opts = self._options
        hass = self.hass

        def state_value(entity_id, cast=float, default=None):
            if not entity_id:
                return default
            state = hass.states.get(entity_id)
            if not state:
                return default
            try:
                return cast(state.state)
            except Exception:
                return state.state

        # Read physical inputs (with fallbacks)
        tdb = state_value(opts.get("sensor_temperatur"), float, 20.0)
        rh = state_value(opts.get("sensor_luftfeuchte"), float, 50.0)
        vr = state_value(opts.get("sensor_wind"), float, 0.2)
        uv = state_value(opts.get("sensor_uv"), float, 0.0)
        sun = state_value(opts.get("sensor_sonnenstrahlung"), float, tdb)
        akt_entity = opts.get("sensor_aktivitaet")
        # Aktivität: if the sensor provides numeric Met use it; if string try mapping common activities
        met = 1.0
        if akt_entity:
            val = state_value(akt_entity, cast=str, default=None)
            if val is None:
                met = 1.0
            else:
                try:
                    met = float(val)
                except Exception:
                    sval = str(val).lower()
                    if "ruh" in sval or "sit" in sval:
                        met = 1.0
                    elif "schlaf" in sval:
                        met = 0.8
                    elif "lang" in sval or "gemüt" in sval:
                        met = 2.0
                    elif "zügig" in sval or "schnell" in sval:
                        met = 3.0
                    else:
                        met = 1.2

        # Persona inputs
        pra = state_value(opts.get("sensor_praeferenz"), cast=str, default="neutral")
        delta_pref = 0.0
        if isinstance(pra, str):
            s = pra.lower()
            if "käl" in s or "kael" in s:
                delta_pref = -0.2  # prefer colder -> target colder (reduce clo slightly)
            elif "wär" in s or "waerm" in s:
                delta_pref = +0.2
            else:
                delta_pref = 0.0

        geschlecht = state_value(opts.get("sensor_geschlecht"), cast=str, default=None)
        alter = state_value(opts.get("sensor_alter"), float, 40.0)
        gewicht = state_value(opts.get("sensor_gewicht"), float, 80.0)
        groesse = state_value(opts.get("sensor_groesse"), float, 1.8)
        kfa = None  # optional

        # For radiation use sun if provided else tdb
        tr = float(sun) if sun is not None else float(tdb)

        # Age alpha heuristic
        alpha_age = 0.0
        try:
            age = float(alter)
            if age >= 65:
                alpha_age = 0.08
            elif age >= 50:
                alpha_age = 0.04
        except Exception:
            alpha_age = 0.0

        # Gender overlay heuristic (small clo offset to be applied as delta_clo_pref)
        delta_clo_pref = 0.0
        if isinstance(geschlecht, str):
            s = geschlecht.lower()
            if "f" in s or "w" in s:
                delta_clo_pref = 0.12

        try:
            clo_req, details = recommend_clo_with_overlays(
                tdb=float(tdb),
                tr=float(tr),
                vr=float(vr),
                rh=float(rh),
                met_tab=float(met),
                height_m=float(groesse),
                mass_kg=float(gewicht),
                alpha_age=float(alpha_age),
                body_fat_perc=kfa,
                k_f=0.007,
                delta_pmv_pref=float(delta_pref),
                delta_clo_pref=float(delta_clo_pref)
            )
            self._state = round(float(clo_req), 3)
            self._attributes = {
                "inputs": opts,
                "details": details,
                "uv_index": uv,
                "last_update": str(self.hass.helpers.event.dt_util.utcnow())
            }
        except Exception as err:
            _LOGGER.exception("Error computing clothing recommendation: %s", err)
            self._state = None
            self._attributes = {"error": str(err), "inputs": opts}

        # write state
        self.async_write_ha_state()

    @property
    def name(self):
        return self._attr_name

    @property
    def state(self):
        return self._state

    @property
    def extra_state_attributes(self):
        return self._attributes