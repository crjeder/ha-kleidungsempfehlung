#!/usr/bin/env python3
"""
CLI for testing the Smart Clothing Recommendation Engine.

Usage:
    python main.py --temp 5 --wind 3 --met 2.0
    python main.py -t 10 -w 5 -m 1.5 --rain 2
    python main.py -t 5 --temp-high 15 -m 1.5    # Temperature range
"""

import argparse
import json
from pathlib import Path

from engine import (
    SmartClothingEngine, Weather, SLOTS,
    get_item_id, is_locked, make_entry
)


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
  python main.py -t 5 --temp-high 15 -m 1.5          # Temperature range: dress for warm, add layers for cold
  python main.py -t 5 --lock torso:1:fleece_midlayer # Lock a favorite item
  python main.py -t 15 -m 2.0 --pmv -0.5             # Prefer cooler (sports)
  python main.py -t 10 -m 1.2 --pmv +0.3             # Prefer warmer
        '''
    )

    # Required parameters
    parser.add_argument('-t', '--temp', type=float, required=True,
                        help='Temperature in °C (or minimum if --temp-high is set)')
    parser.add_argument('--temp-high', type=float, default=None,
                        help='Maximum temperature in °C. If set, optimizes base outfit for warm, then adds layers for cold.')
    parser.add_argument('-w', '--wind', type=float, default=0,
                        help='Wind speed in m/s (default: 0)')
    parser.add_argument('-m', '--met', type=float, default=1.2,
                        help='Metabolic rate in Met units (default: 1.2, seated)')

    # Optional environmental parameters
    parser.add_argument('--rain', type=float, default=0,
                        help='Rain intensity in mm/h (default: 0)')
    parser.add_argument('--humidity', '--rh', type=float, default=50,
                        help='Relative humidity in %% (default: 50)')

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

    # Create engine with solver
    engine = SmartClothingEngine(
        inventory,
        max_layers=args.max_layers,
        solver=args.solver
    )

    # Create weather conditions
    weather = Weather(
        t_ambient=args.temp,
        wind_speed_ms=args.wind,
        rel_humidity=args.humidity,
        rain_mm_h=args.rain
    )

    weather_high = None
    if args.temp_high is not None:
        weather_high = Weather(
            t_ambient=args.temp_high,
            wind_speed_ms=args.wind,
            rel_humidity=args.humidity,
            rain_mm_h=args.rain
        )

    # Process locked items into base_ensemble
    base_ensemble = None
    if args.lock:
        base_ensemble = {slot: [None] * args.max_layers for slot in SLOTS}

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
                base_ensemble[slot][layer] = make_entry(item_id, locked=True)
                print(f"  Locked:       {slot}[{layer}] = {item_id}")
            except ValueError as e:
                print(f"Error: Invalid lock specification '{lock_spec}': {e}")
                return 1

    # Get recommendation
    result = engine.recommend_outfit(
        weather=weather,
        weather_high=weather_high,
        met_rate=args.met,
        pmv_target=args.pmv,
        base_ensemble=base_ensemble
    )

    # Display results
    has_temp_range = weather_high is not None

    print(f"\n--- Input Parameters ---")
    if has_temp_range:
        print(f"  Temperature:  {args.temp:+.1f} to {args.temp_high:+.1f} °C")
    else:
        print(f"  Temperature:  {args.temp:+.1f} °C")
    print(f"  Wind:         {args.wind:.1f} m/s")
    print(f"  Met rate:     {args.met:.1f} Met")
    print(f"  Humidity:     {args.humidity:.0f} %")
    print(f"  Rain:         {args.rain:.1f} mm/h")
    print(f"  Target PMV:   {args.pmv:+.1f}")

    print(f"\n--- Calculation (Fanger PMV Model) ---")
    if has_temp_range:
        print(f"  Target clo (warm): {result.target_clo_warm:.2f} (for {args.temp_high:+.1f}°C)")
        print(f"  Target clo (cold): {result.target_clo:.2f} (for {args.temp:+.1f}°C)")
    else:
        print(f"  Target clo:   {result.target_clo:.2f} (for PMV = {args.pmv:+.1f})")

    print(f"\n--- Recommended Outfit ---")
    print(format_ensemble(result.ensemble, inventory))

    # Calculate PPD based on deviation from target PMV
    pmv_deviation = result.pmv - args.pmv

    print(f"\n--- Result ---")
    print(f"  Achieved clo: {result.achieved_clo:.2f}")
    print(f"  Difference:   {result.achieved_clo - result.target_clo:+.2f}")
    print(f"  PMV:          {result.pmv:+.2f} (target: {args.pmv:+.1f})")
    print(f"  PMV deviation:{pmv_deviation:+.2f}")
    print(f"  PPD:          {result.ppd:.0f}%")

    # Strategy info
    if has_temp_range:
        print(f"  Strategy:     layering (temp range: {args.temp_high - args.temp:.0f}°C)")
    else:
        print(f"  Strategy:     efficiency")

    if args.rain > 1.0:
        print(f"  Note:         Waterproof outer layer required (rain)")

    return 0


if __name__ == '__main__':
    exit(main())
