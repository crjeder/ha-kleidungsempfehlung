"""
Smart Clothing Recommendation Engine

A library for calculating clothing recommendations based on thermal comfort models.
Compatible with any data source (JSON, YAML, database, API, etc.) - the caller
provides inventory data as Python objects.

References:
- ISO 7730: Ergonomics of the thermal environment
- ASHRAE 55: Thermal Environmental Conditions for Human Occupancy
- Fanger, P.O. (1970): Thermal Comfort
- WHO UV Index Guidelines
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pythermalcomfort.models import pmv_ppd_iso
from scipy.optimize import milp, LinearConstraint, Bounds


# =============================================================================
# Fanger PMV Model using pythermalcomfort library (ISO 7730 / ASHRAE 55)
# =============================================================================

def calculate_pmv_ppd(t_air: float, t_radiant: float, v_air: float,
                       rel_humidity: float, met_rate: float, i_cl: float) -> tuple[float, float]:
    """
    Calculate PMV and PPD using the pythermalcomfort library.

    Uses the ISO 7730 implementation from pythermalcomfort for accurate
    Fanger model calculations.

    Args:
        t_air: Air temperature [°C]
        t_radiant: Mean radiant temperature [°C]
        v_air: Air velocity [m/s]
        rel_humidity: Relative humidity [%]
        met_rate: Metabolic rate [Met] (1 Met = 58.2 W/m²)
        i_cl: Clothing insulation [clo]

    Returns:
        Tuple of (PMV, PPD) where:
        - PMV: Predicted Mean Vote [-3 to +3]
        - PPD: Predicted Percentage Dissatisfied [%]
    """
    # Ensure minimum air velocity (still air has some natural convection)
    v_air = max(v_air, 0.1)

    # Call pythermalcomfort's ISO 7730 implementation
    # limit_inputs=False allows outdoor conditions outside ISO comfort zone
    # (ISO limits: 10-30°C, 0-1 m/s wind, 0-2 clo)
    result = pmv_ppd_iso(
        tdb=t_air,
        tr=t_radiant,
        vr=v_air,
        rh=rel_humidity,
        met=met_rate,
        clo=i_cl,
        wme=0.0,  # External work, usually 0
        limit_inputs=False  # Allow outdoor conditions
    )

    pmv = result.pmv
    ppd = result.ppd

    # Clamp PMV to valid range
    pmv = max(-3.0, min(3.0, pmv))
    ppd = max(5.0, min(100.0, ppd))

    return pmv, ppd


def solve_clo_for_pmv_target(t_air: float, t_radiant: float, v_air: float,
                              rel_humidity: float, met_rate: float,
                              pmv_target: float = 0.0,
                              clo_min: float = 0.0, clo_max: float = 4.0,
                              tolerance: float = 0.01) -> float:
    """
    Find the clothing insulation (CLO) that achieves a target PMV value.

    Uses bisection method to find the CLO value where PMV equals the target.

    Args:
        t_air: Air temperature [°C]
        t_radiant: Mean radiant temperature [°C]
        v_air: Air velocity [m/s]
        rel_humidity: Relative humidity [%]
        met_rate: Metabolic rate [Met]
        pmv_target: Target PMV value (default: 0 for thermal neutrality)
        clo_min: Minimum CLO to search [clo]
        clo_max: Maximum CLO to search [clo]
        tolerance: Convergence tolerance for PMV

    Returns:
        Required clothing insulation [clo]
    """
    # Check boundary conditions
    # Higher CLO = warmer body = higher PMV (less negative / more positive)
    # Lower CLO = colder body = lower PMV (more negative)
    pmv_at_min, _ = calculate_pmv_ppd(t_air, t_radiant, v_air, rel_humidity, met_rate, clo_min)
    pmv_at_max, _ = calculate_pmv_ppd(t_air, t_radiant, v_air, rel_humidity, met_rate, clo_max)

    # If target is outside achievable range, return boundary
    if pmv_at_min >= pmv_target:
        # Even with minimum clothing, person feels warm enough (or too warm)
        return clo_min
    if pmv_at_max <= pmv_target:
        # Even with maximum clothing, person still feels cold
        return clo_max

    # Bisection method to find CLO where PMV = target
    clo_low = clo_min
    clo_high = clo_max

    for _ in range(50):  # Max iterations
        clo_mid = (clo_low + clo_high) / 2
        pmv_mid, _ = calculate_pmv_ppd(t_air, t_radiant, v_air, rel_humidity, met_rate, clo_mid)

        if abs(pmv_mid - pmv_target) < tolerance:
            return clo_mid

        # More CLO = warmer = higher PMV
        # If PMV is too high (too warm), reduce CLO
        # If PMV is too low (too cold), increase CLO
        if pmv_mid > pmv_target:
            clo_high = clo_mid  # Too warm, reduce insulation
        else:
            clo_low = clo_mid   # Too cold, increase insulation

    return (clo_low + clo_high) / 2

DEFAULT_MAX_LAYERS = 4
SLOTS = ["head", "neck", "torso", "hands", "legs", "feet"]

# Slot categories for two-phase optimization
COARSE_SLOTS = ["torso", "legs", "feet"]  # Main body - large CLO changes
FINE_SLOTS = ["head", "hands", "neck"]     # Accessories - small CLO changes for fine-tuning

# Required positions that must always be filled (slot, layer_index)
REQUIRED_POSITIONS = [
    ("legs", 0),   # Underwear
    ("feet", 3),   # Shoes (outer layer, since no gaiters in inventory)
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

# Ensemble slot entry: dict with "id" and "locked" fields, or None for empty
EnsembleEntry = dict[str, Any] | None
Ensemble = dict[str, list[EnsembleEntry]]


def get_item_id(entry: EnsembleEntry) -> str | None:
    """Extract item ID from an ensemble entry."""
    if entry is None:
        return None
    return entry.get("id")


def is_locked(entry: EnsembleEntry) -> bool:
    """Check if an ensemble entry is locked."""
    if entry is None:
        return False
    return entry.get("locked", False)


def make_entry(item_id: str | None, locked: bool = False) -> EnsembleEntry:
    """Create an ensemble entry."""
    if item_id is None:
        return None
    return {"id": item_id, "locked": locked}


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

    Uses the Fanger PMV model (ISO 7730) to calculate thermal comfort and
    Integer Linear Programming (ILP) to optimize clothing selection.

    Example:
        engine = SmartClothingEngine(inventory)
        target = engine.calculate_target_clo(t_ambient=8, wind_speed_ms=4, met_rate=2.0)
        outfit = engine.optimize_ensemble_ilp(target, rain_mm_h=0)
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
                              met_rate: float, rel_humidity: float = 50.0,
                              t_radiant: float | None = None,
                              pmv_target: float = 0.0) -> float:
        """
        Calculate the required clothing insulation for thermal comfort using the Fanger PMV model.

        This method uses the full Fanger Predicted Mean Vote (PMV) model from ISO 7730
        to find the clothing insulation that achieves the target PMV value.

        The PMV model accounts for:
        - Air temperature and mean radiant temperature
        - Air velocity (wind speed)
        - Relative humidity
        - Metabolic rate (activity level)
        - Heat exchange through radiation, convection, evaporation, and respiration

        Args:
            t_ambient: Ambient air temperature [°C]
            wind_speed_ms: Wind speed [m/s]
            met_rate: Metabolic rate [Met] (1.0 = seated, 2.0 = walking, 4.0 = running)
            rel_humidity: Relative humidity [%] (default: 50%)
            t_radiant: Mean radiant temperature [°C] (default: same as t_ambient)
            pmv_target: Target PMV value (default: 0.0 for thermal neutrality)
                        Negative = prefer cooler (e.g., -0.5 for sports)
                        Positive = prefer warmer (e.g., +0.5 for cold-sensitive)

        Returns:
            Required clothing insulation [clo] for target PMV

        Note:
            The PMV model naturally accounts for sweating and evaporative cooling
            at higher metabolic rates, so no additional activity correction is needed.
        """
        # Use air temperature as radiant temperature if not specified
        # (reasonable for outdoor conditions without strong solar radiation)
        if t_radiant is None:
            t_radiant = t_ambient

        # Ensure minimum air velocity (still air has some natural convection)
        v_air = max(wind_speed_ms, 0.1)

        # Solve for CLO that gives target PMV
        target_clo = solve_clo_for_pmv_target(
            t_air=t_ambient,
            t_radiant=t_radiant,
            v_air=v_air,
            rel_humidity=rel_humidity,
            met_rate=met_rate,
            pmv_target=pmv_target,
            clo_min=0.0,
            clo_max=4.0,
            tolerance=0.02
        )

        return round(target_clo, 2)

    def calculate_pmv_ppd(self, t_ambient: float, wind_speed_ms: float,
                          met_rate: float, clo: float, rel_humidity: float = 50.0,
                          t_radiant: float | None = None) -> tuple[float, float]:
        """
        Calculate PMV and PPD for given conditions and clothing.

        Useful for evaluating how comfortable a specific outfit will be.

        Args:
            t_ambient: Air temperature [°C]
            wind_speed_ms: Wind speed [m/s]
            met_rate: Metabolic rate [Met]
            clo: Clothing insulation [clo]
            rel_humidity: Relative humidity [%]
            t_radiant: Mean radiant temperature [°C] (default: same as t_ambient)

        Returns:
            Tuple of (PMV, PPD) where:
            - PMV: Predicted Mean Vote [-3 to +3]
            - PPD: Predicted Percentage Dissatisfied [%]
        """
        if t_radiant is None:
            t_radiant = t_ambient

        v_air = max(wind_speed_ms, 0.1)

        pmv, ppd = calculate_pmv_ppd(t_ambient, t_radiant, v_air, rel_humidity, met_rate, clo)

        return round(pmv, 2), round(ppd, 1)

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
            for layer_idx, entry in enumerate(ensemble[slot]):
                item_id = get_item_id(entry)
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
                if get_item_id(ensemble[slot][layer_idx]) is None:
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
                          rain_mm_h: float,
                          base_ensemble: Ensemble | None = None,
                          validate: bool = True) -> Ensemble:
        """
        Core logic for clothing selection using layered slot arrays (heuristic).

        Args:
            target_clo: Target clothing insulation in clo units
            temp_fluctuation: Expected temperature range in °C (higher = prefer layering)
            rain_mm_h: Rain intensity in mm/h (requires waterproof outer on torso if > 1.0)
            base_ensemble: Optional starting ensemble (default: minimal base layers)
            validate: If True, validate base_ensemble before optimization (default: True)

        Returns:
            Optimized ensemble as dict with slot arrays

        Raises:
            ValueError: If validate=True and base_ensemble has validation errors
        """
        inv = self.inventory
        # Initialize ensemble
        if base_ensemble is not None:
            # Validate if requested
            if validate:
                result = self.validate_ensemble(base_ensemble, inv)
                if not result.valid:
                    error_msgs = [f"[{e.slot}:{e.layer}] {e.message}" for e in result.errors]
                    raise ValueError(f"Invalid base_ensemble:\n" + "\n".join(error_msgs))
            ensemble = {slot: list(layers) for slot, layers in base_ensemble.items()}
        else:
            ensemble = self._create_empty_ensemble()
            # Default base layers (fit: base items at index 0)
            ensemble["torso"][0] = make_entry("undershirt_sleeveless")
            ensemble["legs"][0] = make_entry("boxer_synthetic")  # required
            ensemble["feet"][0] = make_entry("socks_standard")
            # Default mid layer (fit: mid items at index 1+)
            ensemble["legs"][1] = make_entry("pants_standard")
            # Shoes as outer layer (no gaiters in inventory)
            ensemble["feet"][self.max_layers - 1] = make_entry("sneakers")  # required

        # Rain requirement: if raining, MUST add waterproof outer layer (unless locked)
        if rain_mm_h > 1.0:
            rain_gear = [i for i in inv
                         if i.get('waterproof')
                         and i.get('slot') == 'torso'
                         and i.get('fit') == 'outer']
            if rain_gear:
                outer_idx = self.max_layers - 1
                outer_entry = ensemble["torso"][outer_idx]
                # Only add rain gear if slot is empty (not if locked with different item)
                if get_item_id(outer_entry) is None:
                    ensemble["torso"][outer_idx] = make_entry(rain_gear[0]["id"])

        # Two-phase optimization:
        # Phase 1 (Balanced): Distribute target CLO evenly across torso, legs, feet
        # Phase 2 (Fine): Use head, hands, neck for fine-tuning

        require_wp = rain_mm_h > 1.0

        # ===== PHASE 1: Balanced optimization (torso, legs, feet) =====
        # Calculate per-slot target: distribute raw CLO evenly across body regions
        # Raw CLO = target_clo / layering_factor (before efficiency loss)
        # Reserve ~10% for accessories (head, hands, neck)
        raw_clo_for_coarse = (target_clo / self.layering_factor) * 0.90
        per_slot_target = raw_clo_for_coarse / len(COARSE_SLOTS)

        # First pass: optimize each slot to balanced target
        for slot in COARSE_SLOTS:
            self._adjust_slot_to_target(ensemble, inv, slot, per_slot_target,
                                        tolerance=0.05)

        # Second pass: redistribute shortfall to high-capacity slots (torso first)
        # Calculate how much CLO we're missing
        current_coarse_clo = sum(self._calc_slot_clo(ensemble, inv, s)
                                  for s in COARSE_SLOTS)
        shortfall = raw_clo_for_coarse - current_coarse_clo

        if shortfall > 0.1:
            # Boost torso to make up the difference (torso has most capacity)
            current_torso = self._calc_slot_clo(ensemble, inv, "torso")
            new_torso_target = current_torso + shortfall
            self._adjust_slot_to_target(ensemble, inv, "torso", new_torso_target,
                                        tolerance=0.05)

            # If still short, try legs
            current_coarse_clo = sum(self._calc_slot_clo(ensemble, inv, s)
                                      for s in COARSE_SLOTS)
            remaining_shortfall = raw_clo_for_coarse - current_coarse_clo
            if remaining_shortfall > 0.1:
                current_legs = self._calc_slot_clo(ensemble, inv, "legs")
                new_legs_target = current_legs + remaining_shortfall
                self._adjust_slot_to_target(ensemble, inv, "legs", new_legs_target,
                                            tolerance=0.05)

        # Handle rain requirement: ensure waterproof outer layer
        if require_wp:
            outer_idx = self.max_layers - 1
            outer_entry = ensemble["torso"][outer_idx]
            outer_id = get_item_id(outer_entry)
            current_outer = self._find_item(inv, outer_id) if outer_id else None
            if current_outer is None or not current_outer.get('waterproof'):
                rain_gear = [i for i in inv
                             if i.get('waterproof')
                             and i.get('slot') == 'torso'
                             and i.get('fit') == 'outer']
                if rain_gear:
                    # Pick waterproof jacket closest to current CLO
                    current_clo = current_outer['clo'] if current_outer else 0
                    best_rain = min(rain_gear, key=lambda x: abs(x['clo'] - current_clo))
                    ensemble["torso"][outer_idx] = make_entry(best_rain['id'])

        # ===== PHASE 2: Fine-tuning with accessories (head, hands, neck) =====
        # First, add base accessories if cold enough and we need more warmth
        current_clo = self._calc_current_clo(ensemble, inv)
        if target_clo - current_clo > 0.05:
            # Add headwear if not present
            if get_item_id(ensemble["head"][0]) is None:
                headwear = [i for i in inv
                            if i.get('slot') == 'head'
                            and i.get('fit') == 'base']
                if headwear:
                    # Pick lightest option first, will upgrade if needed
                    ensemble["head"][0] = make_entry(sorted(headwear, key=lambda x: x['clo'])[0]['id'])

            # Add gloves if not present
            if get_item_id(ensemble["hands"][0]) is None:
                gloves = [i for i in inv
                          if i.get('slot') == 'hands'
                          and i.get('fit') == 'base']
                if gloves:
                    ensemble["hands"][0] = make_entry(sorted(gloves, key=lambda x: x['clo'])[0]['id'])

            # Add neck item if not present
            if get_item_id(ensemble["neck"][0]) is None:
                neck_items = [i for i in inv
                              if i.get('slot') == 'neck'
                              and i.get('fit') == 'base']
                if neck_items:
                    ensemble["neck"][0] = make_entry(sorted(neck_items, key=lambda x: x['clo'])[0]['id'])

        # Fine-tune with accessories
        for _ in range(20):
            current_clo = self._calc_current_clo(ensemble, inv)
            diff = target_clo - current_clo

            if abs(diff) < 0.05:
                break

            if diff > 0:  # Too cold - upgrade accessories
                if not self._upgrade(ensemble, inv, slots=FINE_SLOTS):
                    if not self._add_layer(ensemble, inv, slots=FINE_SLOTS):
                        break  # Can't get closer with accessories
            else:  # Too warm - downgrade/remove accessories
                if not self._downgrade(ensemble, inv, slots=FINE_SLOTS):
                    if not self._remove_optional(ensemble, slots=FINE_SLOTS):
                        break

        return ensemble

    def optimize_ensemble_ilp(self, target_clo: float,
                               rain_mm_h: float = 0.0,
                               base_ensemble: Ensemble | None = None) -> Ensemble:
        """
        Optimize ensemble using Integer Linear Programming (ILP).

        Finds the mathematically optimal clothing combination that minimizes
        the deviation from target_clo while respecting all constraints.

        Args:
            target_clo: Target clothing insulation in clo units
            rain_mm_h: Rain intensity in mm/h (requires waterproof outer on torso if > 1.0)
            base_ensemble: Optional starting ensemble with locked items

        Returns:
            Optimized ensemble as dict with slot arrays

        ILP Formulation:
            Variables: x[i] ∈ {0,1} for each valid (item, slot, layer) placement
            Minimize: |achieved_clo - target_clo| (via slack variables)
            Subject to:
                - Each item used at most once
                - Each position has at most one item
                - Required positions filled
                - Locked items preserved
                - Rain requires waterproof outer layer on torso (assumes rain jacket has hood)
                - Body coverage: 70% of CLO must come from torso + legs
                - Balance: |CLO(torso) - CLO(legs)| <= max(25% of target_clo, 0.1)
                - Decency: underwear (base layer) must be covered by outer layer
                - Layering: outer layer requires mid layer underneath
        """
        inv = self.inventory
        # Build list of valid placements: (item_idx, slot, layer, clo)
        placements: list[tuple[int, str, int, float]] = []

        for item_idx, item in enumerate(inv):
            item_slot = item.get('slot')
            item_fit = item.get('fit', 'mid')
            valid_layers = self.fit_positions.get(item_fit, [])

            if not item_slot:
                continue

            for layer in valid_layers:
                # Check if this slot exists
                if item_slot in SLOTS:
                    placements.append((item_idx, item_slot, layer, item['clo']))

        n_placements = len(placements)
        if n_placements == 0:
            return self._create_empty_ensemble()

        # Variables: x[0:n_placements] binary, s_plus, s_minus continuous (for |clo - target|)
        n_vars = n_placements + 2
        idx_s_plus = n_placements
        idx_s_minus = n_placements + 1

        # Objective: minimize s_plus + s_minus
        c = np.zeros(n_vars)
        c[idx_s_plus] = 1.0
        c[idx_s_minus] = 1.0

        # CLO coefficients with layering factor
        clo_values = np.array([p[3] for p in placements]) * self.layering_factor

        # Separate core constraints (always enforced) from soft constraints (relaxable)
        core_constraints = []
        soft_constraints = []

        # === Equality constraint: sum(clo * x) - s_plus + s_minus = target_clo ===
        A_clo = np.zeros((1, n_vars))
        A_clo[0, :n_placements] = clo_values
        A_clo[0, idx_s_plus] = -1.0
        A_clo[0, idx_s_minus] = 1.0
        soft_constraints.append(LinearConstraint(A_clo, target_clo, target_clo))

        # === Inequality: Each item used at most once ===
        n_items = len(inv)
        A_item = np.zeros((n_items, n_vars))
        for p_idx, (item_idx, slot, layer, clo) in enumerate(placements):
            A_item[item_idx, p_idx] = 1.0
        core_constraints.append(LinearConstraint(A_item, -np.inf, 1.0))

        # === Inequality: Each (slot, layer) has at most one item ===
        slot_layer_pairs = [(s, l) for s in SLOTS for l in range(self.max_layers)]
        A_pos = np.zeros((len(slot_layer_pairs), n_vars))
        for sl_idx, (slot, layer) in enumerate(slot_layer_pairs):
            for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
                if p_slot == slot and p_layer == layer:
                    A_pos[sl_idx, p_idx] = 1.0
        core_constraints.append(LinearConstraint(A_pos, -np.inf, 1.0))

        # === Equality: Required positions must be filled ===
        for req_slot, req_layer in REQUIRED_POSITIONS:
            A_req = np.zeros((1, n_vars))
            for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
                if p_slot == req_slot and p_layer == req_layer:
                    A_req[0, p_idx] = 1.0
            # Only add constraint if there are valid items for this position
            if np.any(A_req[0, :n_placements] > 0):
                core_constraints.append(LinearConstraint(A_req, 1.0, 1.0))

        # === Equality: Locked items must be placed ===
        if base_ensemble:
            for slot in SLOTS:
                for layer_idx, entry in enumerate(base_ensemble[slot]):
                    if is_locked(entry):
                        item_id = get_item_id(entry)
                        # Find placement index for this locked item
                        for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
                            if (inv[item_idx]['id'] == item_id and
                                p_slot == slot and p_layer == layer_idx):
                                A_lock = np.zeros((1, n_vars))
                                A_lock[0, p_idx] = 1.0
                                core_constraints.append(LinearConstraint(A_lock, 1.0, 1.0))
                                break

        # === Inequality: Rain requires waterproof outer layer on torso ===
        if rain_mm_h > 1.0:
            outer_idx = self.max_layers - 1
            A_rain = np.zeros((1, n_vars))
            has_wp_option = False
            for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
                if p_slot == 'torso' and p_layer == outer_idx:
                    item = inv[item_idx]
                    if item.get('waterproof'):
                        A_rain[0, p_idx] = 1.0
                        has_wp_option = True
            # sum(waterproof_outer) >= 1  -->  -sum <= -1
            if has_wp_option:
                core_constraints.append(LinearConstraint(-A_rain, -np.inf, -1.0))

        # === Inequality: Body coverage - 70% of CLO must come from torso + legs ===
        # This ensures practical outfits: shirt/pants before accessories
        # Reduced from 90% to 70% to allow feasibility with limited inventory
        # sum(clo * x for torso/legs items) >= 0.7 * target_clo
        A_body = np.zeros((1, n_vars))
        for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
            if p_slot in ('torso', 'legs'):
                A_body[0, p_idx] = clo_values[p_idx]
        min_body_clo = 0.7 * target_clo
        # sum >= min  -->  -sum <= -min
        soft_constraints.append(LinearConstraint(-A_body, -np.inf, -min_body_clo))

        # === Inequality: Balance - torso and legs should have roughly equal CLO ===
        # |CLO(torso) - CLO(legs)| <= max(0.25 * target_clo, 0.1)
        # This prevents heavy jacket with shorts or vice versa
        # Minimum 0.1 clo keeps balance tight even at low CLO targets
        A_torso = np.zeros(n_vars)
        A_legs = np.zeros(n_vars)
        for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
            if p_slot == 'torso':
                A_torso[p_idx] = clo_values[p_idx]
            elif p_slot == 'legs':
                A_legs[p_idx] = clo_values[p_idx]
        max_diff = max(0.25 * target_clo, 0.1)
        # CLO(torso) - CLO(legs) <= max_diff
        A_balance1 = A_torso - A_legs
        soft_constraints.append(LinearConstraint(A_balance1.reshape(1, -1), -np.inf, max_diff))
        # CLO(legs) - CLO(torso) <= max_diff
        A_balance2 = A_legs - A_torso
        soft_constraints.append(LinearConstraint(A_balance2.reshape(1, -1), -np.inf, max_diff))

        # === Inequality: Underwear must be covered by outer layer ===
        # If base layer (layer 0) is worn, at least one higher layer must be worn
        # sum(x for layer>=1) >= sum(x for layer=0)  -->  sum(layer>=1) - sum(layer=0) >= 0
        for slot in ('torso', 'legs'):
            A_base = np.zeros(n_vars)
            A_outer = np.zeros(n_vars)
            for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
                if p_slot == slot:
                    if p_layer == 0:
                        A_base[p_idx] = 1.0
                    else:
                        A_outer[p_idx] = 1.0
            # sum(outer) - sum(base) >= 0  -->  -(sum(outer) - sum(base)) <= 0
            A_cover = A_base - A_outer
            core_constraints.append(LinearConstraint(A_cover.reshape(1, -1), -np.inf, 0))

        # === Inequality: Mid-layer priority - outer requires mid ===
        # If outer layer (layer 3) is worn, at least one mid layer (layer 1-2) must be worn
        # sum(x for mid layers) >= sum(x for outer layer)
        outer_idx = self.max_layers - 1
        for slot in ('torso', 'legs'):
            A_mid = np.zeros(n_vars)
            A_outer = np.zeros(n_vars)
            for p_idx, (item_idx, p_slot, p_layer, clo) in enumerate(placements):
                if p_slot == slot:
                    if 0 < p_layer < outer_idx:  # mid layers (1, 2)
                        A_mid[p_idx] = 1.0
                    elif p_layer == outer_idx:  # outer layer (3)
                        A_outer[p_idx] = 1.0
            # sum(mid) >= sum(outer)  -->  sum(outer) - sum(mid) <= 0
            A_mid_prio = A_outer - A_mid
            core_constraints.append(LinearConstraint(A_mid_prio.reshape(1, -1), -np.inf, 0))

        # Variable bounds: x ∈ [0,1], s_plus/s_minus ∈ [0, ∞)
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        ub[idx_s_plus] = np.inf
        ub[idx_s_minus] = np.inf
        bounds = Bounds(lb, ub)

        # Integrality: binary for x, continuous for slack
        integrality = np.ones(n_vars, dtype=int)
        integrality[idx_s_plus] = 0
        integrality[idx_s_minus] = 0

        # Solve ILP with all constraints
        all_constraints = core_constraints + soft_constraints
        result = milp(c, constraints=all_constraints, bounds=bounds, integrality=integrality)

        # Fallback: if infeasible, maximize CLO with only core constraints
        if not result.success:
            # New objective: maximize CLO (minimize negative CLO)
            c_max = np.zeros(n_vars)
            c_max[:n_placements] = -clo_values  # Negative = maximize

            result = milp(c_max, constraints=core_constraints, bounds=bounds, integrality=integrality)

        # Build ensemble from solution
        ensemble = self._create_empty_ensemble()

        if result.success:
            x = result.x[:n_placements]
            for p_idx, (item_idx, slot, layer, clo) in enumerate(placements):
                if x[p_idx] > 0.5:  # Binary variable is 1
                    item = inv[item_idx]
                    # Preserve locked status if from base_ensemble
                    locked = False
                    if base_ensemble and is_locked(base_ensemble[slot][layer]):
                        locked = True
                    ensemble[slot][layer] = make_entry(item['id'], locked=locked)

        # Copy locked items from base_ensemble (in case solver didn't include them)
        if base_ensemble:
            for slot in SLOTS:
                for layer_idx, entry in enumerate(base_ensemble[slot]):
                    if is_locked(entry):
                        ensemble[slot][layer_idx] = entry

        return ensemble

    def _calc_current_clo(self, ensemble: Ensemble, inv: Inventory) -> float:
        """Calculate total clo value of current ensemble."""
        total_clo = 0
        for slot in SLOTS:
            total_clo += self._calc_slot_clo(ensemble, inv, slot)
        return total_clo * self.layering_factor

    def _calc_slot_clo(self, ensemble: Ensemble, inv: Inventory, slot: str) -> float:
        """Calculate raw clo value for a single slot (without layering factor)."""
        slot_clo = 0.0
        for entry in ensemble[slot]:
            item_id = get_item_id(entry)
            if item_id is not None:
                item = self._find_item(inv, item_id)
                if item:
                    slot_clo += item['clo']
        return slot_clo

    def _get_used_item_ids(self, ensemble: Ensemble) -> set[str]:
        """Get set of all item IDs currently in the ensemble."""
        used = set()
        for slot in SLOTS:
            for entry in ensemble[slot]:
                item_id = get_item_id(entry)
                if item_id is not None:
                    used.add(item_id)
        return used

    def _adjust_slot_to_target(self, ensemble: Ensemble, inv: Inventory,
                                slot: str, target_slot_clo: float,
                                tolerance: float = 0.1) -> bool:
        """
        Adjust items in a single slot to reach the target CLO value.

        Args:
            ensemble: Current ensemble to modify
            inv: Inventory to select from
            slot: The slot to optimize (e.g., "torso", "legs")
            target_slot_clo: Target raw CLO for this slot (without layering factor)
            tolerance: Acceptable deviation from target (default: 0.1)

        Returns:
            True if any changes were made, False otherwise
        """
        made_change = False

        for _ in range(20):  # Max iterations per slot
            current_slot_clo = self._calc_slot_clo(ensemble, inv, slot)
            diff = target_slot_clo - current_slot_clo

            if abs(diff) < tolerance:
                break

            if diff > 0:  # Need more warmth in this slot
                # First try to add a layer
                if self._add_layer_to_slot(ensemble, inv, slot):
                    made_change = True
                # Then try to upgrade existing items
                elif self._upgrade_slot(ensemble, inv, slot):
                    made_change = True
                else:
                    break  # Can't add more warmth to this slot
            else:  # Need less warmth in this slot
                # First try to downgrade existing items
                if self._downgrade_slot(ensemble, inv, slot):
                    made_change = True
                # Then try to remove a layer
                elif self._remove_from_slot(ensemble, slot):
                    made_change = True
                else:
                    break  # Can't reduce warmth in this slot

        return made_change

    def _add_layer_to_slot(self, ensemble: Ensemble, inv: Inventory, slot: str) -> bool:
        """Add the lightest available layer to an empty position in a slot."""
        # Determine fit types to try based on slot
        if slot == "neck":
            fit_types = ["base"]
        elif slot in ["head", "hands"]:
            # Head and hands can have base (liner/cap) and outer (helmet/gloves)
            fit_types = ["base", "outer"]
        else:
            fit_types = ["mid", "outer", "base"]

        used_ids = self._get_used_item_ids(ensemble)

        for fit in fit_types:
            valid_positions = self._get_valid_positions(fit)
            for layer_idx in valid_positions:
                if get_item_id(ensemble[slot][layer_idx]) is None:
                    options = [i for i in inv
                               if i.get('slot') == slot
                               and i.get('fit', 'mid') == fit
                               and i['id'] not in used_ids]
                    if options:
                        # Add lightest option first
                        lightest = sorted(options, key=lambda x: x['clo'])[0]
                        ensemble[slot][layer_idx] = make_entry(lightest['id'])
                        return True
        return False

    def _upgrade_slot(self, ensemble: Ensemble, inv: Inventory, slot: str) -> bool:
        """Upgrade one item in the slot to a warmer alternative."""
        used_ids = self._get_used_item_ids(ensemble)

        for layer_idx in range(self.max_layers):
            entry = ensemble[slot][layer_idx]
            item_id = get_item_id(entry)

            if item_id is None or is_locked(entry):
                continue

            curr = self._find_item(inv, item_id)
            if curr:
                curr_fit = curr.get('fit', 'mid')
                better = [i for i in inv
                          if i.get('slot') == slot
                          and i.get('fit', 'mid') == curr_fit
                          and i['clo'] > curr['clo']
                          and i['id'] not in used_ids]
                if better:
                    # Pick the next warmer item (smallest increase)
                    next_warmer = sorted(better, key=lambda x: x['clo'])[0]
                    ensemble[slot][layer_idx] = make_entry(next_warmer['id'])
                    return True
        return False

    def _downgrade_slot(self, ensemble: Ensemble, inv: Inventory, slot: str) -> bool:
        """Downgrade one item in the slot to a lighter alternative."""
        used_ids = self._get_used_item_ids(ensemble)

        for layer_idx in range(self.max_layers):
            entry = ensemble[slot][layer_idx]
            item_id = get_item_id(entry)

            if item_id is None or is_locked(entry):
                continue

            curr = self._find_item(inv, item_id)
            if curr:
                curr_fit = curr.get('fit', 'mid')
                lighter = [i for i in inv
                           if i.get('slot') == slot
                           and i.get('fit', 'mid') == curr_fit
                           and i['clo'] < curr['clo']
                           and i['id'] not in used_ids]
                if lighter:
                    # Pick the next lighter item (smallest decrease)
                    next_lighter = sorted(lighter, key=lambda x: -x['clo'])[0]
                    ensemble[slot][layer_idx] = make_entry(next_lighter['id'])
                    return True
        return False

    def _remove_from_slot(self, ensemble: Ensemble, slot: str) -> bool:
        """Remove one optional layer from the slot."""
        # Remove from outer layers first
        for layer_idx in range(self.max_layers - 1, -1, -1):
            if (slot, layer_idx) in REQUIRED_POSITIONS:
                continue
            entry = ensemble[slot][layer_idx]
            if is_locked(entry):
                continue
            if get_item_id(entry) is not None:
                ensemble[slot][layer_idx] = None
                return True
        return False

    def _find_item(self, inv: Inventory, item_id: str) -> InventoryItem | None:
        """Find an item in inventory by ID."""
        return next((i for i in inv if i['id'] == item_id), None)

    def _add_layer(self, ensemble: Ensemble, inv: Inventory,
                    slots: list[str] | None = None) -> bool:
        """Add a new layer to an empty slot position based on fit compatibility.

        Args:
            ensemble: Current ensemble to modify
            inv: Inventory to select from
            slots: Optional list of slots to consider (default: all slots)
        """
        used_ids = self._get_used_item_ids(ensemble)

        add_priority = [
            ("torso", "mid"),
            ("torso", "outer"),
            ("legs", "mid"),
            ("legs", "outer"),
            ("neck", "base"),
            ("head", "base"),
            ("head", "outer"),
            ("hands", "base"),
            ("hands", "outer"),
        ]

        for slot, fit in add_priority:
            # Skip slots not in the allowed list
            if slots is not None and slot not in slots:
                continue
            valid_positions = self._get_valid_positions(fit)
            for layer_idx in valid_positions:
                # Only add to empty positions
                if get_item_id(ensemble[slot][layer_idx]) is None:
                    options = [i for i in inv
                               if i.get('slot') == slot
                               and i.get('fit', 'mid') == fit
                               and i['id'] not in used_ids]
                    if options:
                        ensemble[slot][layer_idx] = make_entry(sorted(options, key=lambda x: x['clo'])[0]['id'])
                        return True
        return False

    def _upgrade(self, ensemble: Ensemble, inv: Inventory,
                  require_waterproof_outer: bool = False,
                  slots: list[str] | None = None) -> bool:
        """Replace an item with a warmer alternative with the same fit.

        Args:
            ensemble: Current ensemble to modify
            inv: Inventory to select from
            require_waterproof_outer: If True, keep waterproof outer layer for rain
            slots: Optional list of slots to consider (default: all slots)

        Locked items are never replaced.
        """
        used_ids = self._get_used_item_ids(ensemble)
        slots_to_check = slots if slots is not None else SLOTS

        for slot in slots_to_check:
            for layer_idx in range(self.max_layers):
                entry = ensemble[slot][layer_idx]
                item_id = get_item_id(entry)

                # Skip empty slots and locked items
                if item_id is None or is_locked(entry):
                    continue

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
                              and i['clo'] > curr['clo']
                              and i['id'] not in used_ids]

                    # If rain requires waterproof, filter for waterproof alternatives
                    if require_waterproof_outer and slot == 'torso' and layer_idx == self.max_layers - 1:
                        better = [i for i in better if i.get('waterproof')]

                    if better:
                        ensemble[slot][layer_idx] = make_entry(sorted(better, key=lambda x: x['clo'])[0]['id'])
                        return True
        return False

    def _remove_optional(self, ensemble: Ensemble,
                          slots: list[str] | None = None) -> bool:
        """
        Remove optional layers (accessories, outer layers) to reduce warmth.

        Called when the ensemble is too warm (achieved_clo > target_clo).
        Items are removed in priority order, simulating how people undress:
        accessories first, then outer layers.

        Args:
            ensemble: Current ensemble to modify
            slots: Optional list of slots to consider (default: all slots)

        Locked items and required positions are never removed.

        Returns:
            True if an item was removed, False if nothing could be removed.
        """
        # Priority order for removal (first = removed first):
        # 1. Accessories (gloves, hat, scarf) - easy to take off
        # 2. Torso layers from outer to mid (index max-1 down to 1)
        # 3. Socks (feet[0])
        # 4. Legs layers from outer to mid (index max-1 down to 1)
        #
        # NOT removed: torso[0] (base layer), REQUIRED_POSITIONS, and locked items
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

        # Filter by allowed slots if specified
        if slots is not None:
            remove_priority = [(s, i) for s, i in remove_priority if s in slots]

        for slot, layer_idx in remove_priority:
            # Don't remove required positions
            if (slot, layer_idx) in REQUIRED_POSITIONS:
                continue
            entry = ensemble[slot][layer_idx]
            # Don't remove locked items
            if is_locked(entry):
                continue
            if get_item_id(entry) is not None:
                ensemble[slot][layer_idx] = None
                return True
        return False

    def _downgrade(self, ensemble: Ensemble, inv: Inventory,
                    slots: list[str] | None = None) -> bool:
        """Replace an item with a lighter alternative with the same fit.

        Args:
            ensemble: Current ensemble to modify
            inv: Inventory to select from
            slots: Optional list of slots to consider (default: all slots)

        Locked items are never replaced.
        """
        used_ids = self._get_used_item_ids(ensemble)
        slots_to_check = slots if slots is not None else SLOTS

        for slot in slots_to_check:
            for layer_idx in range(self.max_layers):
                entry = ensemble[slot][layer_idx]
                item_id = get_item_id(entry)

                # Skip empty slots and locked items
                if item_id is None or is_locked(entry):
                    continue

                curr = self._find_item(inv, item_id)
                if curr:
                    curr_fit = curr.get('fit', 'mid')
                    lighter = [i for i in inv
                               if i.get('slot') == slot
                               and i.get('fit', 'mid') == curr_fit
                               and i['clo'] < curr['clo']
                               and i['id'] not in used_ids]
                    if lighter:
                        ensemble[slot][layer_idx] = make_entry(sorted(lighter, key=lambda x: -x['clo'])[0]['id'])
                        return True
        return False
