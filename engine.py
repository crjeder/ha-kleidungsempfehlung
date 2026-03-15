# Re-export shim — canonical engine lives in the HA component package.
# Loaded directly by file path to avoid triggering the HA __init__.py
# (which has homeassistant/voluptuous dependencies unavailable here).
import importlib.util
import sys
from pathlib import Path

_engine_path = Path(__file__).parent / "custom_components" / "kleidungsempfehlung" / "engine.py"
_spec = importlib.util.spec_from_file_location("_kleidung_engine", _engine_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_kleidung_engine"] = _mod
_spec.loader.exec_module(_mod)

from _kleidung_engine import *  # noqa: F401, F403
