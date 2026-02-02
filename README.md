# Thermischer Komfort & Kleidungsauswahl – Baukasten

**Kurzüberblick (Was ist drin?)**

*   **Bausteine:** UV, Wind, Wetter, Aktivität, Person (Alter, Geschlecht, Größe, Masse, KFA), **Präferenz**
*   **Zielgröße:** erforderliche **clo** (Kleidungsisolation) bei **PMV ≈ 0** (thermisch neutral)
*   **Modellkern:** **ISO 7730/ASHRAE 55 (PMV/PPD)** mit Konvektion/Strahlung/Verdunstung
*   **Overlays:** alters-/geschlechts-/kompositionsbedingte Anpassungen + **Personal Comfort**
*   **Output:** clo‑Empfehlung, Sensitivitäten (RH, Wind, Strahlung), UV‑Schutz

> **Persönliches Empfinden:** Lieber kälter oder wärmer? → auf **kälteste/wärmste** erwartete Situation optimieren.

## Integration für Home Assistant

Diese Custom Integration stellt einen Sensor `sensor.kleidungsempfehlung` bereit, der basierend auf konfigurierbaren Eingangs-Sensoren (z. B. Temperatur, Wind, Luftfeuchte, UV‑Index, Aktivität, Körperdaten) eine Kleidungs-Isolations-Empfehlung (clo) berechnet.

Konfigurierbare Eingangs‑Sensoren (über die Integrations‑UI auswählbar):
- UV (UV‑Index)
- Persönliches_Empfinden (z. B. "kälter", "wärmer", "neutral")
- Geschlecht (string: "male"/"female" optional)
- Alter (Jahre)
- Gewicht (kg)
- Größe (m)
- Wind (m/s)
- Temperatur (°C)
- Luftfeuchtigkeit (%)
- Sonnenstrahlung (optional; °C-Äquivalent / globalstrahlung)

Output des Sensors:
- state: empfohlene clo (float)
- attributes: details (detaillierte Berechnungsdaten), verwendete Eingangs‑Entity-IDs, last_updated, pmv/ppd usw.

Wichtig: Die Integration basiert auf Norm‑Näherungen und Heuristiken aus dem Gist. Für präzise oder normkonforme Anwendungen bitte die ISO/ASHRAE‑Dokumente prüfen.

## Installation

1. Lege das Verzeichnis `custom_components/kleidungsempfehlung` in deinem Home Assistant config‑Verzeichnis an.
2. Kopiere die Dateien aus diesem Repo in das Verzeichnis.
3. Neustart Home Assistant.
4. Füge die Integration über Einstellungen → Geräte & Dienste → Integration hinzufügen → Kleidungsempfehlung hinzu.
5. Wähle die entsprechenden Sensor‑Entities für Temperatur, Wind, RH, etc. in der GUI.

## Hinweise & Weiterentwicklung


### Kleidugnsstück
JSON Schema für Kleidugnsstück
| Feld | Typ | Beschreibung |
|---|---|---|
| id | String | Eindeutiger Name (z.B. heavy_wool_sweater). |
| name | String | Friendly name |
| category | Enum | under_bottom, under_top, torso_base, torso_mid, torso_outer, legs, neck, hands, head, feet, shoes. |
| layer | Enum | base, mid, outer (Wichtig für die Schicht-Logik). |
| clo | Float | Isolationswert im trockenen Zustand. |
| coverage | Enum | short (T-Shirt/Shorts) oder full (Langarm/Lange Hose). |
| waterproof | Bool | true, wenn es als Regenschutz fungiert. |
| windproof | Bool | true, wenn es Wind blockiert (z.B. Hardshell, Windbreaker). |
| coverage | Enum | low, medium, full: the respective coverage of the body area
| upf | Integer | UV Protection Factor
| picture | String | URI to image |
| dscription | String | Short description |
| comment | String | Additional text |

### Person
Eine person hat folgende Slots für Kleidungsstücke
- head
- neck
- torso
- hand
- bottom
- legs
- feet
die in layern übereinander belegt werden können


## Quellen
Siehe Header im eingebetteten Python-Code für Referenzen zu ISO 7730 / ASHRAE 55, WHO UV‑Index, etc.

# Warnung!
Das Repository kann KI-generierte Daten enthalten!
