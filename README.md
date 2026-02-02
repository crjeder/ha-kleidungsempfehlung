# Thermal Comfort & Clothing Selection – Building Blocks

**Quick Overview**

*   **Components:** UV, Wind, Weather, Activity, Person (Age, Gender, Height, Mass, Body Fat), **Preference**
*   **Target:** required **clo** (clothing insulation) at **PMV ≈ 0** (thermally neutral)
*   **Core Model:** **ISO 7730/ASHRAE 55 (PMV/PPD)** with convection/radiation/evaporation
*   **Overlays:** age/gender/body composition adjustments + **Personal Comfort**
*   **Output:** clo recommendation, sensitivities (RH, wind, radiation), UV protection

> **Personal Preference:** Prefer cooler or warmer? → optimize for **coldest/warmest** expected situation.

## Home Assistant Integration

This custom integration provides a sensor `sensor.clothing_recommendation` that calculates a clothing insulation recommendation (clo) based on configurable input sensors (e.g., temperature, wind, humidity, UV index, activity, body data).

Configurable input sensors (selectable via integration UI):
- UV (UV index)
- personal_preference (e.g., "cooler", "warmer", "neutral")
- gender (string: "male"/"female", optional)
- age (years)
- weight (kg)
- height (m)
- wind (m/s)
- temperature (°C)
- humidity (%)
- solar_radiation (optional; perceived temperature increase from direct sun in °C, e.g., +5)

Sensor output:
- state: recommended clo (float)
- attributes: details (detailed calculation data), input entity IDs used, last_updated, pmv/ppd, etc.

Important: This integration is based on standard approximations and heuristics. For precise or standards-compliant applications, please refer to the ISO/ASHRAE documents.

## Installation

1. Create the directory `custom_components/clothing_recommendation` in your Home Assistant config directory.
2. Copy the files from this repo into that directory.
3. Restart Home Assistant.
4. Add the integration via Settings → Devices & Services → Add Integration → Clothing Recommendation.
5. Select the appropriate sensor entities for temperature, wind, RH, etc. in the GUI.

## Notes & Development


### Clothing Item
JSON Schema for Clothing Item
| Field | Type | Description |
|---|---|---|
| id | String | Unique identifier (e.g., heavy_wool_sweater). |
| name | String | Friendly name |
| slot | Enum | head, neck, torso, hands, legs, feet – the body area this item covers. |
| fit | Enum | base, mid, outer – where this item fits in the layering system (see below). |
| clo | Float | Insulation value when dry. |
| waterproof | Bool | true if it functions as rain protection. |
| windproof | Bool | true if it blocks wind (e.g., hardshell, windbreaker). |
| coverage | Enum | low, medium, full: coverage level of the body area (e.g., tank-top=low, t-shirt=medium, long-sleeve=full). |
| upf | Integer | UV Protection Factor |
| picture | String | URI to image |
| description | String | Short description |
| comment | String | Additional text |

#### Fit Values (Layering Suitability)
The `fit` field indicates which layer positions an item is suitable for:

| fit | Suitable Positions | Description |
|-----|-------------------|-------------|
| base | Index 0 only | Worn directly on skin (underwear, base layers, socks) |
| mid | Index 1 to MAX_LAYERS-2 | Insulation layers that can stack (shirts, sweaters, fleece) |
| outer | Index MAX_LAYERS-1 only | Outermost protective layer (jackets, shells, rain gear) |

Example with `MAX_LAYERS = 4`:
- `fit: base` → can only be placed at index 0
- `fit: mid` → can be placed at index 1 or 2
- `fit: outer` → can only be placed at index 3

### Person
A person has the following slots for clothing items. Each slot is an array of length `MAX_LAYERS = 4` (index 0 = base, 1 = mid, 2 = outer, 3 = shell):

| Slot | Description |
|------|-------------|
| head | Headwear (beanie, hat, etc.) |
| neck | Neck/collar (scarf, buff, etc.) |
| torso | Upper body (undershirt → shirt → sweater → jacket) |
| hands | Hands (liner → gloves → mittens) |
| legs | Legs (underwear → pants → overpants) |
| feet | Feet (socks → shoes → gaiters) |

Example:
```json
{
  "torso": ["t_shirt_synthetic", "fleece_mid", "hardshell", null],
  "legs": ["boxer_synthetic", "pants_standard", null, null]
}
```


## References
See header in embedded Python code for references to ISO 7730 / ASHRAE 55, WHO UV Index, etc.

# Warning!
This repository may contain AI-generated data!
