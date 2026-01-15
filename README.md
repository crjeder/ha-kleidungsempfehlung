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

- Erweiterungen: Ensemble‑Optimierer, UV‑Schutz‑Warnings, Logging von Nutzer-Feedback für Kalibrierung (PCM).
- Tests: Die enthaltenen `__main__`-Blöcke in der Engine sind einfache Selbsttests; nutze sie lokal zur Validierung.

## Quellen
Siehe Header im eingebetteten Python-Code für Referenzen zu ISO 7730 / ASHRAE 55, WHO UV‑Index, etc.

# Warnung!
Das Repository kann KI-generierte Daten enthalten!
