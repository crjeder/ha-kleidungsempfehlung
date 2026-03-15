# Kleidungsempfehlung — Claude Code Guide

## Project structure
```
ha-kleidungsempfehlung/
├── custom_components/kleidungsempfehlung/   ← integration source (live bind-mounted into HA)
├── ha-test/
│   ├── docker-compose.yml                  ← HA test container
│   ├── ha-config/
│   │   ├── configuration.yaml              ← minimal HA config with stub sensors
│   │   └── inventory.yaml                  ← test clothing inventory
│   └── scripts/
│       ├── validate.sh                     ← MAIN ENTRY POINT for validation
│       ├── check_sensors.py                ← sensor spec checker (called by validate.sh)
│       └── restart.sh                      ← restart HA + tail logs
└── .ha_token                               ← Long-Lived Access Token (never commit this)
```

## Workflow after code changes

1. **Edit** code in `custom_components/kleidungsempfehlung/`
   - No deploy step needed — directory is bind-mounted into the container.

2. **Restart HA** to reload the integration:
   ```bash
   ./ha-test/scripts/restart.sh
   ```
   Watch for errors in the log output (integration logs at DEBUG level).

3. **Run full validation:**
   ```bash
   HA_TOKEN=$(cat .ha_token) ./ha-test/scripts/validate.sh
   ```
   - Exit 0 = all checks passed
   - Exit 1 = see output for failures

## What validate.sh checks
1. Container is running and HA API is reachable
2. No ERROR/EXCEPTION lines in `custom_components.kleidungsempfehlung` logs
3. `sensor.kleidungsempfehlung_test` exists and has a numeric clo state
4. Required attributes present: `ensemble`, `target_clo`, `achieved_clo`, `pmv`, `ppd`, `weather`, `met_rate`
5. Ensemble dict is non-empty
6. PMV in [-3, 3], PPD in [5, 100]
7. weather.temperature is plausible

## First-time setup (do once)
```bash
# Start the container
cd ha-test && docker compose up -d

# Wait ~60s for first boot, then open HA and create a Long-Lived Access Token:
# Profile → Security → Long-lived access tokens → Create Token
echo "<your_token>" > ../.ha_token
chmod 600 ../.ha_token

# Run validation
HA_TOKEN=$(cat .ha_token) ./scripts/validate.sh
```

## Tear down / reset
```bash
cd ha-test && docker compose down -v
```

## Stub sensor entity IDs (defined in ha-config/configuration.yaml)
| Entity | Purpose |
|--------|---------|
| `input_number.test_temperature` | Außentemperatur (°C) |
| `input_number.test_humidity` | Luftfeuchte (%) |
| `input_number.test_wind` | Wind (m/s) |
| `input_number.test_rain` | Regen (mm/h) |
| `input_number.test_alter` | Alter |
| `input_select.test_aktivitaet` | Aktivitätslevel |
| `input_select.test_geschlecht` | Geschlecht |

These can be changed via the HA developer tools UI or REST API to test different scenarios.

## Dependencies (installed automatically by HA)
- pythermalcomfort >= 2.8.0
- scipy >= 1.10.0
- numpy >= 1.24.0
