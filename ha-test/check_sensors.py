#!/usr/bin/env python3
"""
check_sensors.py - Validates kleidungsempfehlung sensor output against spec.
Usage: python3 check_sensors.py <ha_url> <token>
"""

import sys
import json
import urllib.request
import urllib.error

def get_states(ha_url: str, token: str) -> dict:
    req = urllib.request.Request(
        f"{ha_url}/api/states",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return {s["entity_id"]: s for s in json.loads(resp.read())}


def check_sensor(states: dict, entity_id: str) -> dict | None:
    return states.get(entity_id)


def validate(ha_url: str, token: str) -> bool:
    try:
        states = get_states(ha_url, token)
    except Exception as e:
        print(f"❌ Could not fetch states from HA API: {e}")
        return False

    failed = False
    sensor_id = "sensor.kleidungsempfehlung_test"   # matches name in configuration.yaml

    sensor = check_sensor(states, sensor_id)
    if not sensor:
        print(f"❌ Sensor '{sensor_id}' not found. Available sensors:")
        for eid in sorted(states.keys()):
            if "kleidung" in eid.lower():
                print(f"   {eid} = {states[eid]['state']}")
        return False

    state = sensor["state"]
    attrs = sensor.get("attributes", {})

    print(f"Sensor: {sensor_id}")
    print(f"  state (clo):  {state}")
    print(f"  unit:         {attrs.get('unit_of_measurement', 'n/a')}")
    print(f"  icon:         {attrs.get('icon', 'n/a')}")
    print()

    # --- Spec checks ---

    # State must be a valid float (clo value)
    try:
        clo = float(state)
        if 0.0 <= clo <= 10.0:
            print(f"✅ state is valid clo value: {clo}")
        else:
            print(f"❌ state clo={clo} is out of plausible range [0, 10]")
            failed = True
    except (ValueError, TypeError):
        if state in ("unavailable", "unknown", "None"):
            print(f"❌ Sensor state is '{state}' — integration may have failed to load or compute")
        else:
            print(f"❌ state '{state}' is not a float")
        failed = True

    # Required attributes
    required_attrs = ["ensemble", "target_clo", "achieved_clo", "pmv", "ppd", "weather", "met_rate"]
    for attr in required_attrs:
        if attr in attrs:
            print(f"✅ attribute '{attr}' present: {attrs[attr]}")
        else:
            print(f"❌ attribute '{attr}' missing")
            failed = True

    # ensemble must be a dict with at least one slot
    if "ensemble" in attrs:
        ensemble = attrs["ensemble"]
        if isinstance(ensemble, dict) and len(ensemble) > 0:
            print(f"✅ ensemble has {len(ensemble)} slot(s): {list(ensemble.keys())}")
        else:
            print(f"❌ ensemble is empty or wrong type: {ensemble}")
            failed = True

    # PMV must be in realistic range [-3, 3]
    if "pmv" in attrs:
        try:
            pmv = float(attrs["pmv"])
            if -3.0 <= pmv <= 3.0:
                print(f"✅ PMV={pmv} is in valid range [-3, 3]")
            else:
                print(f"⚠️  PMV={pmv} is outside typical range [-3, 3]")
        except (ValueError, TypeError):
            print(f"❌ PMV value not numeric: {attrs['pmv']}")
            failed = True

    # PPD must be [5, 100]
    if "ppd" in attrs:
        try:
            ppd = float(attrs["ppd"])
            if 5.0 <= ppd <= 100.0:
                print(f"✅ PPD={ppd} is in valid range [5, 100]")
            else:
                print(f"⚠️  PPD={ppd} is outside expected range [5, 100]")
        except (ValueError, TypeError):
            print(f"❌ PPD value not numeric: {attrs['ppd']}")
            failed = True

    # weather sub-dict should reflect our stub sensor values (roughly)
    if "weather" in attrs:
        w = attrs["weather"]
        temp = w.get("temperature")
        if temp is not None:
            try:
                t = float(temp)
                if -30 <= t <= 50:
                    print(f"✅ weather.temperature={t}°C looks plausible")
                else:
                    print(f"⚠️  weather.temperature={t} looks implausible")
            except (ValueError, TypeError):
                print(f"❌ weather.temperature is not numeric: {temp}")
                failed = True

    print()
    if failed:
        print("❌ VALIDATION FAILED — see above")
        return False
    else:
        print("✅ ALL CHECKS PASSED")
        return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <ha_url> <token>")
        sys.exit(1)
    ok = validate(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
