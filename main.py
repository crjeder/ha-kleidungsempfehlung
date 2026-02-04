#!/usr/bin/env python3
"""
CLI for testing the Smart Clothing Recommendation Engine.

Usage:
    python main.py --temp 5 --wind 3 --met 2.0
    python main.py -t 10 -w 5 -m 1.5 --rain 2
"""

import argparse
import json
from math import exp
from pathlib import Path

from engine import SmartClothingEngine, SLOTS, REQUIRED_POSITIONS, get_item_id, is_locked, make_entry


def calculate_ppd(pmv: float) -> float:
    """
    Calculate Predicted Percentage Dissatisfied (PPD) from PMV.

    PPD = 100 - 95 × exp(-0.03353×PMV⁴ - 0.2179×PMV²)
    """
    ppd = 100 - 95 * exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return max(5.0, min(100.0, ppd))


def load_inventory(path: str = "inventory.json") -> list[dict]:
    """Load inventory from JSON file."""
    inventory_path = Path(__file__).parent / path
    with open(inventory_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_ensemble(ensemble: dict, inventory: list[dict]) -> str:
    """Format ensemble for display. Shows locked items with a lock symbol."""
    lines = []
    inv_lookup = {item['id']: item for item in inventory}

    for slot in SLOTS:
        layers = ensemble.get(slot, [])
        items = []
        for idx, entry in enumerate(layers):
            item_id = get_item_id(entry)
            if item_id:
                item = inv_lookup.get(item_id, {})
                name = item.get('name', item_id)
                clo = item.get('clo', 0)
                lock_icon = "[L]" if is_locked(entry) else ""
                items.append(f"[{idx}]{lock_icon} {name} ({clo:.2f} clo)")

        if items:
            lines.append(f"  {slot:6}: {', '.join(items)}")
        else:
            lines.append(f"  {slot:6}: (empty)")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Smart Clothing Recommendation Engine CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --temp 5 --wind 3 --met 2.0
  python main.py -t 10 -w 5 -m 1.5 --rain 2
  python main.py -t -5 -w 8 -m 3.0 --fluctuation 15
  python main.py -t 5 --lock torso:1:fleece_midlayer  # Lock a favorite item
  python main.py -t 15 -m 2.0 --pmv -0.5             # Prefer cooler (sports)
  python main.py -t 10 -m 1.2 --pmv +0.3             # Prefer warmer
        '''
    )

    # Required parameters
    parser.add_argument('-t', '--temp', type=float, required=True,
                        help='Ambient temperature in °C')
    parser.add_argument('-w', '--wind', type=float, default=0,
                        help='Wind speed in m/s (default: 0)')
    parser.add_argument('-m', '--met', type=float, default=1.2,
                        help='Metabolic rate in Met units (default: 1.2, seated)')

    # Optional environmental parameters
    parser.add_argument('--rain', type=float, default=0,
                        help='Rain intensity in mm/h (default: 0)')
    parser.add_argument('--humidity', '--rh', type=float, default=50,
                        help='Relative humidity in %% (default: 50)')
    parser.add_argument('--fluctuation', type=float, default=5,
                        help='Expected temperature fluctuation in °C (default: 5)')

    # Comfort parameters
    parser.add_argument('--pmv', type=float, default=0.0,
                        help='Target PMV value (default: 0 = neutral). '
                             'Negative = prefer cooler (e.g., -0.5 for sports), '
                             'Positive = prefer warmer (e.g., +0.5 for cold-sensitive)')

    # Engine parameters
    parser.add_argument('--max-layers', type=int, default=4,
                        help='Maximum layers per slot (default: 4)')
    parser.add_argument('--inventory', type=str, default='inventory.json',
                        help='Path to inventory JSON file')

    # Locked items
    parser.add_argument('-l', '--lock', action='append', metavar='SLOT:LAYER:ITEM',
                        help='Lock an item in place (can be used multiple times). '
                             'Format: slot:layer:item_id (e.g., torso:1:fleece_midlayer)')

    # Solver options
    parser.add_argument('--solver', choices=['ilp', 'heuristic'], default='ilp',
                        help='Optimization solver: ilp (optimal) or heuristic (fast). Default: ilp')

    # Output options
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed output')

    args = parser.parse_args()

    # Load inventory
    try:
        inventory = load_inventory(args.inventory)
        print(f"Loaded {len(inventory)} items from {args.inventory}")
    except FileNotFoundError:
        print(f"Error: Inventory file '{args.inventory}' not found")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in inventory file: {e}")
        return 1

    # Create engine
    engine = SmartClothingEngine(inventory, max_layers=args.max_layers)

    # Calculate target clo using Fanger PMV model
    target_clo = engine.calculate_target_clo(
        t_ambient=args.temp,
        wind_speed_ms=args.wind,
        met_rate=args.met,
        rel_humidity=args.humidity,
        pmv_target=args.pmv
    )

    print(f"\n--- Input Parameters ---")
    print(f"  Temperature:  {args.temp:+.1f} °C")
    print(f"  Wind:         {args.wind:.1f} m/s")
    print(f"  Met rate:     {args.met:.1f} Met")
    print(f"  Humidity:     {args.humidity:.0f} %")
    print(f"  Rain:         {args.rain:.1f} mm/h")
    print(f"  Fluctuation:  {args.fluctuation:.0f} °C")
    print(f"  Target PMV:   {args.pmv:+.1f}")

    print(f"\n--- Calculation (Fanger PMV Model) ---")
    print(f"  Target clo:   {target_clo:.2f} (for PMV = {args.pmv:+.1f})")

    # Process locked items into base_ensemble
    base_ensemble = None
    if args.lock:
        # Start with default base layers (same as engine default)
        base_ensemble = {slot: [None] * args.max_layers for slot in SLOTS}
        base_ensemble["torso"][0] = make_entry("undershirt_sleeveless")
        base_ensemble["legs"][0] = make_entry("boxer_synthetic")
        base_ensemble["feet"][0] = make_entry("socks_standard")
        base_ensemble["legs"][1] = make_entry("pants_standard")
        base_ensemble["feet"][1] = make_entry("sneakers")

        # Apply locked items on top
        for lock_spec in args.lock:
            try:
                parts = lock_spec.split(':')
                if len(parts) != 3:
                    print(f"Error: Invalid lock format '{lock_spec}'. Use slot:layer:item_id")
                    return 1
                slot, layer_str, item_id = parts
                layer = int(layer_str)
                if slot not in SLOTS:
                    print(f"Error: Unknown slot '{slot}'. Valid: {', '.join(SLOTS)}")
                    return 1
                if layer < 0 or layer >= args.max_layers:
                    print(f"Error: Layer {layer} out of range (0-{args.max_layers-1})")
                    return 1
                # Create locked entry
                base_ensemble[slot][layer] = make_entry(item_id, locked=True)
                print(f"  Locked:       {slot}[{layer}] = {item_id}")
            except ValueError as e:
                print(f"Error: Invalid lock specification '{lock_spec}': {e}")
                return 1

    # Optimize ensemble
    if args.solver == 'ilp':
        ensemble = engine.optimize_ensemble_ilp(
            target_clo=target_clo,
            rain_mm_h=args.rain,
            base_ensemble=base_ensemble
        )
    else:
        ensemble = engine.optimize_ensemble(
            target_clo=target_clo,
            temp_fluctuation=args.fluctuation,
            rain_mm_h=args.rain,
            base_ensemble=base_ensemble
        )

    # Calculate actual clo
    actual_clo = engine._calc_current_clo(ensemble, inventory)

    print(f"\n--- Recommended Outfit ---")
    print(format_ensemble(ensemble, inventory))

    # Calculate PMV/PPD for the achieved outfit
    pmv, _ = engine.calculate_pmv_ppd(
        t_ambient=args.temp,
        wind_speed_ms=args.wind,
        met_rate=args.met,
        clo=actual_clo,
        rel_humidity=args.humidity
    )

    # Calculate PPD based on deviation from target PMV
    # This way, hitting the target exactly gives PPD = 5% (minimum)
    pmv_deviation = pmv - args.pmv
    ppd = calculate_ppd(pmv_deviation)

    print(f"\n--- Result ---")
    print(f"  Achieved clo: {actual_clo:.2f}")
    print(f"  Difference:   {actual_clo - target_clo:+.2f}")
    print(f"  PMV:          {pmv:+.2f} (target: {args.pmv:+.1f})")
    print(f"  PMV deviation:{pmv_deviation:+.2f}")
    print(f"  PPD:          {ppd:.0f}% (from target)")

    # Strategy info
    strategy = "layering" if args.fluctuation >= 7 else "efficiency"
    print(f"  Strategy:     {strategy}")

    if args.rain > 1.0:
        print(f"  Note:         Waterproof outer layer required (rain)")

    return 0


if __name__ == '__main__':
    exit(main())
