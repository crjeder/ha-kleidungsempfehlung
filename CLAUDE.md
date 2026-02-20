# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **Smart Clothing Recommendation Engine** based on the Fanger PMV thermal comfort model (ISO 7730/ASHRAE 55). It has two components:

1. **Standalone Python library + CLI** (`engine.py`, `main.py`) — usable independently for testing
2. **Home Assistant custom integration** (`custom_components/kleidungsempfehlung/`) — wraps the engine as a HA sensor

## Running the CLI

```bash
# Install dependencies first
pip install pythermalcomfort scipy numpy

# Basic usage
python main.py --temp 5 --wind 3 --met 2.0

# With temperature range (layering strategy: dress for warm, add layers for cold)
python main.py -t 5 --temp-high 15 -m 1.5

# Lock a specific item at a layer position
python main.py -t 5 --lock torso:1:fleece_midlayer

# Target a specific PMV (negative = prefer cooler, positive = warmer)
python main.py -t 15 -m 2.0 --pmv -0.5

# Use heuristic solver instead of ILP
python main.py --temp 10 --solver heuristic

# Custom inventory file
python main.py --temp 10 --inventory my_wardrobe.json
```

## Architecture

### Core Engine (`engine.py`)

The engine is a standalone library with no HA dependency. Key flow:

1. **PMV solver**: `solve_clo_for_pmv_target()` uses bisection to find the CLO value achieving a target PMV (default: 0 = neutral). Delegates to `pythermalcomfort.models.pmv_ppd_iso`.

2. **`SmartClothingEngine`** class orchestrates everything:
   - `calculate_target_clo()` → calls PMV solver
   - `recommend_outfit()` → main entry point; handles single-temp and temp-range modes
   - `optimize_ensemble_ilp()` → ILP (default solver, uses `scipy.optimize.milp`)
   - `optimize_ensemble()` → heuristic two-phase greedy solver

3. **Temperature range mode**: When `weather_high` is provided, Phase 1 optimizes for the warm temperature (minimal clothing), then Phase 2 locks that outfit and adds layers for the cold temperature.

4. **Layering factor**: Effective CLO = raw CLO × 0.835 (accounts for reduced insulation when layers compress).

### Data Model

**Clothing item** (in `inventory.json` / `inventory.yaml`):
- `slot`: one of `head`, `neck`, `torso`, `hands`, `legs`, `feet`
- `fit`: `base` (layer 0 only), `mid` (layers 1..MAX-2), `outer` (layer MAX-1 only)
- `clo`: insulation value

**Ensemble**: `dict[slot → list[entry | None]]` where entry = `{"id": str, "locked": bool}`. Helpers: `get_item_id()`, `is_locked()`, `make_entry()`.

**Required positions** (always filled): `legs[0]` (underwear) and `feet[3]` (shoes).

### ILP Formulation (`optimize_ensemble_ilp`)

Binary variables x[i] for each valid (item, slot, layer) placement. Minimizes |achieved_clo - target_clo| via slack variables. Constraints (core + soft):
- Each item used at most once; each (slot, layer) position has at most one item
- Required positions filled; locked items preserved
- Rain → waterproof outer on torso
- Body coverage: ≥70% of CLO from torso + legs
- Balance: |CLO(torso) - CLO(legs)| ≤ max(0.25×target, 0.1)
- Decency: base layer must be covered; outer layer requires a mid layer

On infeasibility, falls back to maximizing CLO with only core constraints.

### Home Assistant Integration (`custom_components/kleidungsempfehlung/`)

- **`__init__.py`**: Validates YAML config with voluptuous schemas; stores config in `hass.data[DOMAIN]`
- **`sensor.py`**: `KleidungsempfehlungSensor` — polls HA state machine for weather/person entity values, calls `engine.recommend_outfit()`, writes achieved CLO as state (unit: `clo`)
- **`engine.py`**: Copy of the standalone engine (kept in sync manually)
- **`config_flow.py`**: UI-based config flow for HA (alternative to YAML)
- **`const.py`**: All constant strings (CONF_*, DEFAULT_* values, DOMAIN)

**Configuration is primarily YAML-based** (not UI flow). Example in `custom_components/kleidungsempfehlung/example_configuration.yaml`. Inventory can be referenced with `!include inventory.yaml`.

**Key sensor attributes**: `ensemble`, `target_clo`, `achieved_clo`, `pmv`, `ppd`, `weather`, `met_rate`, `pmv_target`.

## Important Notes

- `engine.py` in root and `custom_components/kleidungsempfehlung/engine.py` must be kept in sync — the HA integration uses its own copy.
- `inventory.json` (used by CLI) and `inventory.yaml` (used by HA via `!include`) contain the same data in different formats.
- The `fit` field determines which layer index is valid. With `max_layers=4`: base→[0], mid→[1,2], outer→[3].
- Activity sensor in HA accepts either numeric Met values or German/English text keywords (mapped in `sensor.py:_get_met_rate()`).
