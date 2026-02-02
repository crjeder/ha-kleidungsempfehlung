import json
import itertools

class SmartClothingEngine:
    def __init__(self, inventory_file):
        with open(inventory_file, 'r') as f:
            self.inventory = json.load(f)
        self.layering_factor = 0.835
        self.t_skin_comfort = 33.5

    def calculate_target_clo(self, t_ambient, wind_speed_ms, met_rate):
        """Berechnet den physikalisch benötigten Clo-Wert inkl. Windchill."""
        # Windchill-Effekt auf die Grenzschicht der Luft
        v_kmh = wind_speed_ms * 3.6
        if t_ambient < 15 and v_kmh > 5:
            t_eff = 13.12 + 0.6215 * t_ambient - 11.37 * (v_kmh**0.16) + 0.3965 * t_ambient * (v_kmh**0.16)
        else:
            t_eff = t_ambient
        
        # I_req Formel nach ISO 7730 Grundzügen
        i_req = (self.t_skin_comfort - t_eff) / (0.155 * met_rate) - 0.1 # 0.1 als Pauschale für Luftschicht
        return max(0.1, round(i_req, 2))

    def filter_inventory(self, uv_index, rain_mm_h):
        """Filtert und bewertet das Inventar nach UV und Regen."""
        filtered = []
        for item in self.inventory:
            # UV-Logik: Bevorzuge lange Kleidung bei hohem UV
            if uv_index >= 6 and item['coverage'] == 'short':
                continue # Überspringe kurze Sachen bei extremer Sonne
            
            # Regen-Logik: Markiere Items für Eskalation
            temp_item = item.copy()
            if rain_mm_h > 0.5:
                if item['material'] == 'cotton':
                    temp_item['clo'] *= 0.2 # Baumwolle isoliert nass fast gar nicht
            
            filtered.append(temp_item)
        return filtered

    def optimize_ensemble(self, target_clo, temp_fluctuation, filtered_inv, rain_mm_h):
        """Kern-Logik zur Auswahl der Kleidung."""
        # Strategie-Wahl
        strategy = "layering" if temp_fluctuation >= 7 else "efficiency"
        
        # Grund-Ensemble definieren (Starter-Set)
        ensemble = {
            "under_bottom": "boxer_short_synthetic", # Empfehlung bei Sport/Outdoor
            "torso_base": "t_shirt_synthetic",
            "legs": "pants_standard",
            "feet": "socks_standard"
        }

        # Regen-Zwang: Wenn es regnet, MUSS eine wasserdichte Schicht dazu
        if rain_mm_h > 1.0:
            rain_gear = [i for i in filtered_inv if i.get('waterproof')]
            if rain_gear:
                ensemble["torso_outer"] = rain_gear[0]["id"]

        # Iterative Anpassung
        for _ in range(15):
            current_clo = self._calc_current_clo(ensemble, filtered_inv)
            diff = target_clo - current_clo

            if abs(diff) < 0.05: break

            if diff > 0: # Zu kalt
                if strategy == "layering":
                    if not self._add_layer(ensemble, filtered_inv): self._upgrade(ensemble, filtered_inv)
                else:
                    if not self._upgrade(ensemble, filtered_inv): self._add_layer(ensemble, filtered_inv)
            else: # Zu warm
                if not self._remove_optional(ensemble): self._downgrade(ensemble, filtered_inv)

        return ensemble

    def _calc_current_clo(self, ensemble, inv):
        ids = ensemble.values()
        return sum(next(i['clo'] for i in inv if i['id'] == eid) for eid in ids) * self.layering_factor

    def _add_layer(self, ensemble, inv):
        slots = ["torso_mid", "torso_outer", "neck", "head", "hands"]
        for s in slots:
            if s not in ensemble:
                options = [i for i in inv if i['category'] == s]
                if options:
                    ensemble[s] = sorted(options, key=lambda x: x['clo'])[0]['id']
                    return True
        return False

    def _upgrade(self, ensemble, inv):
        for s, eid in ensemble.items():
            curr = next(i for i in inv if i['id'] == eid)
            better = [i for i in inv if i['category'] == s and i['clo'] > curr['clo']]
            if better:
                ensemble[s] = sorted(better, key=lambda x: x['clo'])[0]['id']
                return True
        return False

    def _remove_optional(self, ensemble):
        for s in ["neck", "hands", "head", "torso_outer"]:
            if s in ensemble:
                del ensemble[s]; return True
        return False

    def _downgrade(self, ensemble, inv):
        for s, eid in ensemble.items():
            curr = next(i for i in inv if i['id'] == eid)
            lighter = [i for i in inv if i['category'] == s and i['clo'] < curr['clo']]
            if lighter:
                ensemble[s] = sorted(lighter, key=lambda x: x['clo'])[0]['id']
                return True
        return False

# --- Anwendung ---
# engine = SmartClothingEngine('inventory.json')
# target = engine.calculate_target_clo(t_ambient=8, wind_speed_ms=4, met_rate=150)
# final_outfit = engine.optimize_ensemble(target, temp_fluctuation=10, ...)