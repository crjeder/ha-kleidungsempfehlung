# CLAUDE.md

## Project Overview

A **Smart Clothing Recommendation Engine** based on the Fanger PMV thermal comfort model (ISO 7730/ASHRAE 55). Two components:

1. **Standalone Python library + CLI** (`engine.py`, `main.py`) — usable independently
2. **Home Assistant custom integration** (`custom_components/kleidungsempfehlung/`) — wraps engine as a HA sensor

## CLI

```bash
pip install pythermalcomfort scipy numpy

python main.py --temp 5 --wind 3 --met 2.0
python main.py -t 5 --temp-high 15 -m 1.5          # temperature range / layering
python main.py -t 5 --lock torso:1:fleece_midlayer  # lock item at position
python main.py -t 15 -m 2.0 --pmv -0.5             # target PMV (neg=cooler)
python main.py --temp 10 --solver heuristic         # heuristic instead of ILP
python main.py --temp 10 --inventory my_wardrobe.json
```

## Testing (HA Integration)

```bash
./ha-test/restart.sh                               # restart HA + tail logs
HA_TOKEN=$(cat .ha_token) ./ha-test/validate.sh    # full validation (exit 0 = pass)
```

See [ha-test/CLAUDE.md](ha-test/CLAUDE.md) for first-time Docker setup, validate.sh checks, and stub sensor details.

## Architecture

### Core Engine (`engine.py`)

1. **PMV solver**: `solve_clo_for_pmv_target()` — bisection to find CLO achieving target PMV. Delegates to `pythermalcomfort.models.pmv_ppd_iso`.
2. **`SmartClothingEngine`** orchestrates:
   - `calculate_target_clo()` → PMV solver
   - `recommend_outfit()` → main entry point; single-temp and temp-range modes
   - `optimize_ensemble_ilp()` → ILP solver (`scipy.optimize.milp`)
   - `optimize_ensemble()` → heuristic two-phase greedy solver
3. **Temperature range mode**: Phase 1 optimizes for warm temp (minimal clothing), Phase 2 locks that outfit and adds layers for cold temp.
4. **Layering factor**: Effective CLO = raw CLO × 0.835 (compression reduces insulation).

### Data Model

**Clothing item** (`inventory.json` / `inventory.yaml`):
- `slot`: `head`, `neck`, `torso`, `hands`, `legs`, `feet`
- `fit`: `base` (layer 0 only) | `mid` (layers 1..MAX-2) | `outer` (layer MAX-1 only)
- `clo`: insulation value

**Ensemble**: `dict[slot → list[entry | None]]` where entry = `{"id": str, "locked": bool}`. Helpers: `get_item_id()`, `is_locked()`, `make_entry()`.

**Required positions**: `legs[0]` (underwear) and `feet[3]` (shoes).

### ILP Formulation (`optimize_ensemble_ilp`)

Binary variables x[i] per valid (item, slot, layer). Minimizes |achieved_clo - target_clo| via slack variables, with constraints for coverage, balance, decency, rain-proofing, and locked items. Falls back to maximizing CLO on infeasibility.

### Home Assistant Integration (`custom_components/kleidungsempfehlung/`)

- **`__init__.py`**: Validates YAML config (voluptuous); stores in `hass.data[DOMAIN]`
- **`sensor.py`**: `KleidungsempfehlungSensor` — polls HA state, calls `engine.recommend_outfit()`, writes achieved CLO as state (unit: `clo`)
- **`engine.py`**: Copy of standalone engine (kept in sync manually)
- **`config_flow.py`**: UI config flow (alternative to YAML)
- **`const.py`**: All `CONF_*` / `DEFAULT_*` constants

Config is **YAML-based**. Example: `custom_components/kleidungsempfehlung/example_configuration.yaml`. Inventory via `!include inventory.yaml`.

**Key sensor attributes**: `ensemble`, `target_clo`, `achieved_clo`, `pmv`, `ppd`, `weather`, `met_rate`, `pmv_target`.

## Gotchas

- `engine.py` root and `custom_components/kleidungsempfehlung/engine.py` must stay in sync — edit both.
- `inventory.json` (CLI) and `inventory.yaml` (HA) are the same data in different formats.
- `fit` determines valid layer indices. With `max_layers=4`: base→[0], mid→[1,2], outer→[3].
- Activity sensor accepts numeric Met values or German/English text keywords (mapped in `sensor.py:_get_met_rate()`).

## Session Protocol

### Start
1. Read `claude-progress.txt`
2. Read `git log --oneline -10`
3. Work on exactly ONE task

### End
1. Update `README.md` if relevant (follow https://www.makeareadme.com/)
2. Update `claude-progress.txt`
3. Document changes in `CLAUDE.md` (learning from mistakes)
4. Update `TODO.md`
5. Document in `CHANGELOG.md` (https://keepachangelog.com/en/1.1.0/)
6. `git commit` with descriptive message
