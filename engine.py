"""
Smart Clothing Recommendation Engine

A library for calculating clothing recommendations based on thermal comfort models.
Compatible with any data source (JSON, YAML, database, API, etc.) - the caller
provides inventory data as Python objects.

References:
- ISO 7730: Ergonomics of the thermal environment
- ASHRAE 55: Thermal Environmental Conditions for Human Occupancy
- WHO UV Index Guidelines
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

DEFAULT_MAX_LAYERS = 4
SLOTS = ["head", "neck", "torso", "hands", "legs", "feet"]

# Required positions that must always be filled (slot, layer_index)
REQUIRED_POSITIONS = [
    ("legs", 0),   # Underwear
    ("feet", 1),   # Shoes
]


def compute_fit_positions(max_layers: int) -> dict[str, list[int]]:
    """
    Compute valid layer positions for each fit type.

    Args:
        max_layers: Total number of layers in the system

    Returns:
        Dict mapping fit type to list of valid layer indices
    """
    if max_layers < 2:
        raise ValueError("max_layers must be at least 2 (base + outer)")

    return {
        "base": [0],                                    # Only index 0 (directly on skin)
        "mid": list(range(1, max_layers - 1)),          # Index 1 to max_layers-2 (insulation)
        "outer": [max_layers - 1],                      # Only last index (protective shell)
    }

# Type aliases for clarity
InventoryItem = dict[str, Any]
Inventory = list[InventoryItem]
Ensemble = dict[str, list[str | None]]


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Critical: ensemble cannot be used
    WARNING = "warning"  # Non-critical: ensemble may work but has issues


@dataclass
class ValidationIssue:
    """A single validation issue found in an ensemble."""
    severity: ValidationSeverity
    slot: str | None
    layer: int | None
    item_id: str | None
    message: str


@dataclass
class ValidationResult:
    """Result of ensemble validation."""
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class SmartClothingEngine:
    """
    Engine for calculating clothing recommendations based on environmental conditions.

    The engine accepts inventory data as Python objects, making it compatible with
    any data source (JSON, YAML, database, API, Home Assistant entities, etc.).

    Example usage:
        # Load inventory from any source
        import json
        with open('inventory.json') as f:
            inventory = json.load(f)

        # Or from YAML
        import yaml
        with open('inventory.yaml') as f:
            inventory = yaml.safe_load(f)

        # Or from a database, API, etc.
        inventory = my_database.get_clothing_items()

        # Create engine with loaded data
        engine = SmartClothingEngine(inventory)
        target = engine.calculate_target_clo(t_ambient=8, wind_speed_ms=4, met_rate=2.0)
        filtered = engine.filter_inventory(uv_index=3, rain_mm_h=0)
        outfit = engine.optimize_ensemble(target, temp_fluctuation=10,
                                          filtered_inv=filtered, rain_mm_h=0)
    """

    def __init__(self, inventory: Inventory,
                 max_layers: int = DEFAULT_MAX_LAYERS,
                 layering_factor: float = 0.835,
                 t_skin_comfort: float = 33.5):
        """
        Initialize the clothing engine with inventory data.

        Args:
            inventory: List of clothing item dictionaries. Each item should have
                       at minimum: id, slot, fit, clo. See README for full schema.
            max_layers: Number of layers per slot (default: 4). Minimum is 2.
            layering_factor: Efficiency factor for layered clothing (default: 0.835).
                            Accounts for reduced insulation when layers overlap.
            t_skin_comfort: Target skin temperature for thermal comfort in °C
                           (default: 33.5°C based on ISO 7730).
        """
        self.inventory = inventory
        self.max_layers = max_layers
        self.fit_positions = compute_fit_positions(max_layers)
        self.layering_factor = layering_factor
        self.t_skin_comfort = t_skin_comfort

    def calculate_target_clo(self, t_ambient: float, wind_speed_ms: float,
                              met_rate: float) -> float:
        """
        Calculate the physically required clo value including windchill.

        Args:
            t_ambient: Ambient air temperature in °C
            wind_speed_ms: Wind speed in m/s
            met_rate: Metabolic rate in Met units (1.0 = seated, 2.0 = walking)

        Returns:
            Required clothing insulation in clo units
        """
        # Windchill effect on the air boundary layer
        v_kmh = wind_speed_ms * 3.6
        if t_ambient < 15 and v_kmh > 5:
            t_eff = (13.12 + 0.6215 * t_ambient
                     - 11.37 * (v_kmh ** 0.16)
                     + 0.3965 * t_ambient * (v_kmh ** 0.16))
        else:
            t_eff = t_ambient

        # I_req formula based on ISO 7730 principles
        # met_rate is in Met units, 1 Met = 58.2 W/m²
        # R_cl (clothing thermal resistance) = 0.155 m²K/W per clo
        met_watts = met_rate * 58.2  # Convert Met to W/m²
        i_req = (self.t_skin_comfort - t_eff) / (0.155 * met_watts) - 0.1
        return max(0.1, round(i_req, 2))

    def filter_inventory(self, uv_index: float, rain_mm_h: float) -> Inventory:
        """
        Filter and evaluate inventory based on UV and rain conditions.

        Args:
            uv_index: Current UV index (0-11+)
            rain_mm_h: Rain intensity in mm/h

        Returns:
            Filtered inventory with adjusted clo values for wet conditions
        """
        filtered = []
        for item in self.inventory:
            # UV logic: prefer full coverage clothing at high UV
            if uv_index >= 6 and item.get('coverage') == 'low':
                continue  # Skip low coverage items in extreme sun

            # Rain logic: non-waterproof clothing loses insulation when wet
            temp_item = item.copy()
            if rain_mm_h > 0.5 and not item.get('waterproof', False):
                temp_item['clo'] *= 0.3  # Wet clothing insulates significantly worse

            filtered.append(temp_item)
        return filtered

    def _create_empty_ensemble(self) -> Ensemble:
        """Create an empty ensemble with all slots initialized to None arrays."""
        return {slot: [None] * self.max_layers for slot in SLOTS}

    def validate_ensemble(self, ensemble: Ensemble,
                          inv: Inventory | None = None) -> ValidationResult:
        """
        Validate an ensemble for correctness.

        Checks performed:
        - Structure: all required slots present with correct array length
        - Item existence: all item IDs exist in inventory
        - Slot match: item's slot field matches ensemble slot
        - Fit match: item's fit allows placement at layer index
        - No duplicates: same item ID not used multiple times

        Args:
            ensemble: The ensemble to validate
            inv: Inventory to validate against (defaults to self.inventory)

        Returns:
            ValidationResult with valid=True if no errors, plus list of issues
        """
        if inv is None:
            inv = self.inventory

        issues: list[ValidationIssue] = []
        seen_ids: set[str] = set()

        # Check structure: all slots present
        for slot in SLOTS:
            if slot not in ensemble:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    slot=slot, layer=None, item_id=None,
                    message=f"Missing slot '{slot}' in ensemble"
                ))
                continue

            # Check array length
            if len(ensemble[slot]) != self.max_layers:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    slot=slot, layer=None, item_id=None,
                    message=f"Slot '{slot}' has {len(ensemble[slot])} layers, expected {self.max_layers}"
                ))
                continue

            # Check each layer position
            for layer_idx, item_id in enumerate(ensemble[slot]):
                if item_id is None:
                    continue

                # Check for duplicates
                if item_id in seen_ids:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        slot=slot, layer=layer_idx, item_id=item_id,
                        message=f"Duplicate item '{item_id}' used multiple times"
                    ))
                seen_ids.add(item_id)

                # Check item exists in inventory
                item = self._find_item(inv, item_id)
                if item is None:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        slot=slot, layer=layer_idx, item_id=item_id,
                        message=f"Item '{item_id}' not found in inventory"
                    ))
                    continue

                # Check slot match
                item_slot = item.get('slot')
                if item_slot and item_slot != slot:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        slot=slot, layer=layer_idx, item_id=item_id,
                        message=f"Item '{item_id}' has slot '{item_slot}', placed in '{slot}'"
                    ))

                # Check fit match
                if not self._item_fits_position(item, layer_idx):
                    item_fit = item.get('fit', 'mid')
                    valid_positions = self.fit_positions.get(item_fit, [])
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        slot=slot, layer=layer_idx, item_id=item_id,
                        message=f"Item '{item_id}' has fit '{item_fit}' (valid: {valid_positions}), "
                                f"placed at index {layer_idx}"
                    ))

        # Check for unknown slots in ensemble
        for slot in ensemble:
            if slot not in SLOTS:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    slot=slot, layer=None, item_id=None,
                    message=f"Unknown slot '{slot}' in ensemble (will be ignored)"
                ))

        # Check required positions are filled
        for slot, layer_idx in REQUIRED_POSITIONS:
            if slot in ensemble and layer_idx < len(ensemble[slot]):
                if ensemble[slot][layer_idx] is None:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        slot=slot, layer=layer_idx, item_id=None,
                        message=f"Required position {slot}[{layer_idx}] is empty"
                    ))

        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        return ValidationResult(valid=not has_errors, issues=issues)

    def _item_fits_position(self, item: InventoryItem, layer_idx: int) -> bool:
        """Check if an item's fit allows it to be placed at the given layer index."""
        fit = item.get('fit', 'mid')
        allowed_positions = self.fit_positions.get(fit, [])
        return layer_idx in allowed_positions

    def _get_valid_positions(self, fit: str) -> list[int]:
        """Get list of valid layer indices for a given fit type."""
        return self.fit_positions.get(fit, self.fit_positions['mid'])

    def optimize_ensemble(self, target_clo: float, temp_fluctuation: float,
                          filtered_inv: Inventory, rain_mm_h: float,
                          base_ensemble: Ensemble | None = None,
                          validate: bool = True) -> Ensemble:
        """
        Core logic for clothing selection using layered slot arrays.

        Args:
            target_clo: Target clothing insulation in clo units
            temp_fluctuation: Expected temperature range in °C (higher = prefer layering)
            filtered_inv: Pre-filtered inventory from filter_inventory()
            rain_mm_h: Rain intensity in mm/h
            base_ensemble: Optional starting ensemble (default: minimal base layers)
            validate: If True, validate base_ensemble before optimization (default: True)

        Returns:
            Optimized ensemble as dict with slot arrays

        Raises:
            ValueError: If validate=True and base_ensemble has validation errors
        """
        # Strategy selection
        strategy = "layering" if temp_fluctuation >= 7 else "efficiency"

        # Initialize ensemble
        if base_ensemble is not None:
            # Validate if requested
            if validate:
                result = self.validate_ensemble(base_ensemble, filtered_inv)
                if not result.valid:
                    error_msgs = [f"[{e.slot}:{e.layer}] {e.message}" for e in result.errors]
                    raise ValueError(f"Invalid base_ensemble:\n" + "\n".join(error_msgs))
            ensemble = {slot: list(layers) for slot, layers in base_ensemble.items()}
        else:
            ensemble = self._create_empty_ensemble()
            # Default base layers (fit: base items at index 0)
            ensemble["torso"][0] = "t_shirt_base"        # base layer
            ensemble["legs"][0] = "boxer_synthetic"      # base layer (required)
            ensemble["feet"][0] = "socks_standard"       # base layer
            # Default mid layer (fit: mid items at index 1+)
            ensemble["legs"][1] = "pants_standard"       # mid layer
            ensemble["feet"][1] = "sneakers"             # shoes (required)

        # Rain requirement: if raining, MUST add waterproof outer layer
        if rain_mm_h > 1.0:
            rain_gear = [i for i in filtered_inv
                         if i.get('waterproof')
                         and i.get('slot') == 'torso'
                         and i.get('fit') == 'outer']
            if rain_gear:
                outer_idx = self.max_layers - 1
                if ensemble["torso"][outer_idx] is None:
                    ensemble["torso"][outer_idx] = rain_gear[0]["id"]

        # Cold weather: add head and hand coverage when target_clo is high
        # target_clo > 1.3 roughly corresponds to temperatures below ~10°C
        if target_clo > 1.3:
            # Add headwear if not present
            if ensemble["head"][0] is None:
                headwear = [i for i in filtered_inv
                            if i.get('slot') == 'head'
                            and i.get('fit') == 'base']
                if headwear:
                    # Pick warmest option for cold weather
                    ensemble["head"][0] = sorted(headwear, key=lambda x: -x['clo'])[0]['id']

            # Add gloves if not present
            if ensemble["hands"][0] is None:
                gloves = [i for i in filtered_inv
                          if i.get('slot') == 'hands'
                          and i.get('fit') == 'base']
                if gloves:
                    ensemble["hands"][0] = sorted(gloves, key=lambda x: -x['clo'])[0]['id']

        # Iterative adjustment (more iterations for complex ensembles)
        for _ in range(50):
            current_clo = self._calc_current_clo(ensemble, filtered_inv)
            diff = target_clo - current_clo

            if abs(diff) < 0.05:
                break

            if diff > 0:  # Too cold
                require_wp = rain_mm_h > 1.0
                if strategy == "layering":
                    if not self._add_layer(ensemble, filtered_inv):
                        self._upgrade(ensemble, filtered_inv, require_waterproof_outer=require_wp)
                else:
                    if not self._upgrade(ensemble, filtered_inv, require_waterproof_outer=require_wp):
                        self._add_layer(ensemble, filtered_inv)
            else:  # Too warm
                if not self._remove_optional(ensemble):
                    self._downgrade(ensemble, filtered_inv)

        return ensemble

    def _calc_current_clo(self, ensemble: Ensemble, inv: Inventory) -> float:
        """Calculate total clo value of current ensemble."""
        total_clo = 0
        for slot in SLOTS:
            for item_id in ensemble[slot]:
                if item_id is not None:
                    item = next((i for i in inv if i['id'] == item_id), None)
                    if item:
                        total_clo += item['clo']
        return total_clo * self.layering_factor

    def _find_item(self, inv: Inventory, item_id: str) -> InventoryItem | None:
        """Find an item in inventory by ID."""
        return next((i for i in inv if i['id'] == item_id), None)

    def _add_layer(self, ensemble: Ensemble, inv: Inventory) -> bool:
        """Add a new layer to an empty slot position based on fit compatibility."""
        add_priority = [
            ("torso", "mid"),
            ("torso", "outer"),
            ("neck", "base"),
            ("head", "base"),
            ("hands", "base"),
        ]

        for slot, fit in add_priority:
            valid_positions = self._get_valid_positions(fit)
            for layer_idx in valid_positions:
                if ensemble[slot][layer_idx] is None:
                    options = [i for i in inv
                               if i.get('slot') == slot
                               and i.get('fit', 'mid') == fit]
                    if options:
                        ensemble[slot][layer_idx] = sorted(options, key=lambda x: x['clo'])[0]['id']
                        return True
        return False

    def _upgrade(self, ensemble: Ensemble, inv: Inventory,
                  require_waterproof_outer: bool = False) -> bool:
        """Replace an item with a warmer alternative with the same fit."""
        for slot in SLOTS:
            for layer_idx in range(self.max_layers):
                item_id = ensemble[slot][layer_idx]
                if item_id is not None:
                    curr = self._find_item(inv, item_id)
                    if curr:
                        # Don't replace waterproof outer layer when rain requires it
                        if (require_waterproof_outer and
                            slot == 'torso' and
                            layer_idx == self.max_layers - 1 and
                            curr.get('waterproof')):
                            continue

                        curr_fit = curr.get('fit', 'mid')
                        better = [i for i in inv
                                  if i.get('slot') == slot
                                  and i.get('fit', 'mid') == curr_fit
                                  and i['clo'] > curr['clo']]

                        # If rain requires waterproof, filter for waterproof alternatives
                        if require_waterproof_outer and slot == 'torso' and layer_idx == self.max_layers - 1:
                            better = [i for i in better if i.get('waterproof')]

                        if better:
                            ensemble[slot][layer_idx] = sorted(better, key=lambda x: x['clo'])[0]['id']
                            return True
        return False

    def _remove_optional(self, ensemble: Ensemble) -> bool:
        """
        Remove optional layers (accessories, outer layers) to reduce warmth.

        Called when the ensemble is too warm (achieved_clo > target_clo).
        Items are removed in priority order, simulating how people undress:
        accessories first, then outer layers.

        Returns:
            True if an item was removed, False if nothing could be removed.
        """
        # Priority order for removal (first = removed first):
        # 1. Accessories (gloves, hat, scarf) - easy to take off
        # 2. Torso layers from outer to mid (index max-1 down to 1)
        # 3. Socks (feet[0])
        # 4. Legs layers from outer to mid (index max-1 down to 1)
        #
        # NOT removed: torso[0] (base layer), and REQUIRED_POSITIONS
        # (legs[0]=underwear, feet[1]=shoes)
        remove_priority: list[tuple[str, int]] = [
            ("hands", 0),
            ("head", 0),
            ("neck", 0),
        ]
        # Add torso layers from outer inward (skip base layer at index 0)
        for i in range(self.max_layers - 1, 0, -1):
            remove_priority.append(("torso", i))
        # Socks
        remove_priority.append(("feet", 0))
        # Add legs layers from outer inward (skip base layer at index 0)
        for i in range(self.max_layers - 1, 0, -1):
            remove_priority.append(("legs", i))
        # remove shirt
        remove_priority.append(("torso", 0))
        
        for slot, layer_idx in remove_priority:
            # Don't remove required positions
            if (slot, layer_idx) in REQUIRED_POSITIONS:
                continue
            if ensemble[slot][layer_idx] is not None:
                ensemble[slot][layer_idx] = None
                return True
        return False

    def _downgrade(self, ensemble: Ensemble, inv: Inventory) -> bool:
        """Replace an item with a lighter alternative with the same fit."""
        for slot in SLOTS:
            for layer_idx in range(self.max_layers):
                item_id = ensemble[slot][layer_idx]
                if item_id is not None:
                    curr = self._find_item(inv, item_id)
                    if curr:
                        curr_fit = curr.get('fit', 'mid')
                        lighter = [i for i in inv
                                   if i.get('slot') == slot
                                   and i.get('fit', 'mid') == curr_fit
                                   and i['clo'] < curr['clo']]
                        if lighter:
                            ensemble[slot][layer_idx] = sorted(lighter, key=lambda x: -x['clo'])[0]['id']
                            return True
        return False
