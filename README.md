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


## Thermal Calculation Model (Fanger PMV)

This engine uses the **Fanger Predicted Mean Vote (PMV) model** from ISO 7730 [1] to calculate the required clothing insulation for thermal comfort.

### The PMV Model

The PMV model predicts thermal sensation on a 7-point scale:

| PMV | Thermal Sensation |
|-----|-------------------|
| -3 | Cold |
| -2 | Cool |
| -1 | Slightly cool |
| **0** | **Neutral (target)** |
| +1 | Slightly warm |
| +2 | Warm |
| +3 | Hot |

The model finds the clothing insulation (CLO) that achieves **PMV = 0** (thermal neutrality).

### Heat Balance Equation

The PMV model is based on human thermal balance:

```
H = Q_skin + Q_respiration + Q_radiation + Q_convection
```

Where:
- **H** = Internal heat production [W/m²] = M - W (metabolic rate minus external work)
- **Q_skin** = Heat loss through skin (diffusion + sweating)
- **Q_respiration** = Heat loss through breathing (latent + dry)
- **Q_radiation** = Radiative heat exchange with surroundings
- **Q_convection** = Convective heat exchange with air

### Key Equations

**Skin heat loss (diffusion):**
```
Q_diff = 3.05×10⁻³ × (5733 - 6.99×H - Pa)
```

**Skin heat loss (sweating, if H > 58.15 W/m²):**
```
Q_sweat = 0.42 × (H - 58.15)
```

**Respiratory heat loss:**
```
Q_resp = 1.7×10⁻⁵ × M × (5867 - Pa) + 0.0014 × M × (34 - Ta)
```

**Radiative heat loss:**
```
Q_rad = 3.96×10⁻⁸ × fcl × [(Tcl + 273)⁴ - (Tr + 273)⁴]
```

**Convective heat loss:**
```
Q_conv = fcl × hc × (Tcl - Ta)
```

**Clothing area factor:**
```
fcl = 1.00 + 1.290×Icl    for Icl ≤ 0.078 m²K/W
fcl = 1.05 + 0.645×Icl    for Icl > 0.078 m²K/W
```

**Convective heat transfer coefficient:**
```
hc = max(2.38×|Tcl - Ta|^0.25, 12.1×√v)
```

**PMV calculation:**
```
PMV = (0.303 × e^(-0.036×M) + 0.028) × L
```

Where **L** is the thermal load (heat production minus heat losses).

### Predicted Percentage Dissatisfied (PPD)

PPD predicts the percentage of people who would feel thermally uncomfortable:

```
PPD = 100 - 95 × e^(-0.03353×PMV⁴ - 0.2179×PMV²)
```

| PMV | PPD |
|-----|-----|
| 0 | 5% (minimum) |
| ±0.5 | 10% |
| ±1.0 | 26% |
| ±2.0 | 77% |

### Input Parameters

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Air temperature | Ta | °C | Ambient air temperature |
| Radiant temperature | Tr | °C | Mean radiant temperature (default: Ta) |
| Air velocity | v | m/s | Wind speed |
| Relative humidity | RH | % | Relative humidity (default: 50%) |
| Metabolic rate | M | Met | Activity level (1 Met = 58.2 W/m²) |
| Clothing insulation | Icl | clo | Clothing thermal resistance |

### Metabolic Rates (Met)

| Activity | Met | W/m² |
|----------|-----|------|
| Seated, relaxed | 1.0 | 58.2 |
| Standing, light work | 1.2 | 69.8 |
| Walking 4 km/h | 2.0 | 116.4 |
| Walking 5 km/h | 2.6 | 151.3 |
| Running 10 km/h | 4.0 | 232.8 |

Source: ISO 8996 [5]

### Layering Factor

When multiple clothing layers are worn, the effective insulation is reduced due to compression and air gap reduction [4]:

```
CLO_effective = CLO_total × λ
```

Where **λ = 0.835** (approximately 83.5% efficiency for layered clothing).

### Two-Phase Optimization Algorithm

The clothing selection uses a balanced two-phase optimization:

**Phase 1 - Balanced Distribution (Torso, Legs, Feet):**
```
Raw_CLO_target = (Target_CLO / λ) × 0.90
Per_slot_target = Raw_CLO_target / 3
```

Each body region receives an equal share of the target insulation, creating realistic combinations.

**Phase 2 - Fine-tuning (Head, Hands, Neck):**
Accessories are added/adjusted to reach the final target within ±0.05 clo tolerance.

### Example Calculations

**Scenario: Walking at 15°C, 50% RH, Met=2.0**
```
Input: Ta=15°C, Tr=15°C, v=0.1m/s, RH=50%, M=2.0 Met
PMV solver finds: Icl = 0.88 clo for PMV ≈ 0
Result: PMV = -0.06, PPD = 5%
```

**Scenario: Sitting at 20°C, 50% RH, Met=1.0**
```
Input: Ta=20°C, Tr=20°C, v=0.1m/s, RH=50%, M=1.0 Met
PMV solver finds: Icl = 1.50 clo for PMV ≈ 0
Result: PMV = -0.04, PPD = 5%
```

**Scenario: Walking at -15°C, 50% RH, Met=2.0**
```
Input: Ta=-15°C, Tr=-15°C, v=0.1m/s, RH=50%, M=2.0 Met
PMV solver finds: Icl = 3.62 clo for PMV ≈ 0
Result: PMV = -0.01, PPD = 5%
```

## References

[1] **ISO 7730:2005** - Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices and local thermal comfort criteria. International Organization for Standardization.

[1a] **Fanger, P.O. (1970)** - Thermal Comfort: Analysis and Applications in Environmental Engineering. Danish Technical Press, Copenhagen. (Original work establishing the PMV/PPD model)

[2] **Osczevski, R. & Bluestein, M. (2005)** - The new wind chill equivalent temperature chart. Bulletin of the American Meteorological Society, 86(10), 1453-1458. https://doi.org/10.1175/BAMS-86-10-1453

[3] **Nikolopoulou, M. & Steemers, K. (2003)** - Thermal comfort and psychological adaptation as a guide for designing urban spaces. Energy and Buildings, 35(1), 95-101. https://doi.org/10.1016/S0378-7788(02)00084-1

[4] **Havenith, G. (2002)** - The interaction between clothing insulation and thermoregulation. Exogenous Dermatology, 1(5), 221-230. https://doi.org/10.1159/000068802

[5] **ISO 8996:2004** - Ergonomics of the thermal environment — Determination of metabolic rate. International Organization for Standardization.

[6] **ASHRAE Standard 55-2020** - Thermal Environmental Conditions for Human Occupancy. American Society of Heating, Refrigerating and Air-Conditioning Engineers.

[7] **WHO (2002)** - Global Solar UV Index: A Practical Guide. World Health Organization. ISBN 92-4-159007-6

# Warning!
This repository may contain AI-generated data!
