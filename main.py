#!/usr/bin/env python3
"""
CLI for testing the Smart Clothing Recommendation Engine.

Usage:
    python main.py --temp 5 --wind 3 --met 2.0
    python main.py -t 10 -w 5 -m 1.5 --uv 6 --rain 2
"""

import argparse
import json
from pathlib import Path

from engine import SmartClothingEngine, SLOTS


def load_inventory(path: str = "inventory.json") -> list[dict]:
    """Load inventory from JSON file."""
    inventory_path = Path(__file__).parent / path
    with open(inventory_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_ensemble(ensemble: dict, inventory: list[dict], filtered_inv: list[dict] | None = None) -> str:
    """Format ensemble for display."""
    lines = []

    # Create lookup dicts
    inv_lookup = {item['id']: item for item in inventory}
    filtered_lookup = {item['id']: item for item in (filtered_inv or inventory)}

    for slot in SLOTS:
        layers = ensemble.get(slot, [])
        items = []
        for idx, item_id in enumerate(layers):
            if item_id:
                item = inv_lookup.get(item_id, {})
                filtered_item = filtered_lookup.get(item_id, item)
                name = item.get('name', item_id)
                orig_clo = item.get('clo', 0)
                eff_clo = filtered_item.get('clo', orig_clo)

                if abs(orig_clo - eff_clo) > 0.01:
                    items.append(f"[{idx}] {name} ({eff_clo:.2f}/{orig_clo:.2f} clo)")
                else:
                    items.append(f"[{idx}] {name} ({orig_clo:.2f} clo)")

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
  python main.py -t 10 -w 5 -m 1.5 --uv 6 --rain 2
  python main.py -t -5 -w 8 -m 3.0 --fluctuation 15
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
    parser.add_argument('--uv', type=float, default=0,
                        help='UV index (default: 0)')
    parser.add_argument('--rain', type=float, default=0,
                        help='Rain intensity in mm/h (default: 0)')
    parser.add_argument('--fluctuation', type=float, default=5,
                        help='Expected temperature fluctuation in °C (default: 5)')

    # Engine parameters
    parser.add_argument('--max-layers', type=int, default=4,
                        help='Maximum layers per slot (default: 4)')
    parser.add_argument('--inventory', type=str, default='inventory.json',
                        help='Path to inventory JSON file')

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

    # Calculate target clo
    target_clo = engine.calculate_target_clo(
        t_ambient=args.temp,
        wind_speed_ms=args.wind,
        met_rate=args.met
    )

    print(f"\n--- Input Parameters ---")
    print(f"  Temperature:  {args.temp:+.1f} °C")
    print(f"  Wind:         {args.wind:.1f} m/s")
    print(f"  Met rate:     {args.met:.1f} Met")
    print(f"  UV index:     {args.uv:.0f}")
    print(f"  Rain:         {args.rain:.1f} mm/h")
    print(f"  Fluctuation:  {args.fluctuation:.0f} °C")

    print(f"\n--- Calculation ---")
    print(f"  Target clo:   {target_clo:.2f}")

    # Filter inventory
    filtered_inv = engine.filter_inventory(
        uv_index=args.uv,
        rain_mm_h=args.rain
    )

    if args.verbose:
        print(f"  Filtered:     {len(filtered_inv)}/{len(inventory)} items available")

    # Optimize ensemble
    ensemble = engine.optimize_ensemble(
        target_clo=target_clo,
        temp_fluctuation=args.fluctuation,
        filtered_inv=filtered_inv,
        rain_mm_h=args.rain
    )

    # Calculate actual clo
    actual_clo = engine._calc_current_clo(ensemble, filtered_inv)

    print(f"\n--- Recommended Outfit ---")
    print(format_ensemble(ensemble, inventory, filtered_inv))

    print(f"\n--- Result ---")
    print(f"  Achieved clo: {actual_clo:.2f}")
    print(f"  Difference:   {actual_clo - target_clo:+.2f}")

    # Strategy info
    strategy = "layering" if args.fluctuation >= 7 else "efficiency"
    print(f"  Strategy:     {strategy}")

    if args.rain > 1.0:
        print(f"  Note:         Waterproof outer layer required (rain)")
    if args.uv >= 6:
        print(f"  Note:         High UV - prefer full coverage clothing")

    return 0


if __name__ == '__main__':
    exit(main())
