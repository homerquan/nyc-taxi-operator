"""Run a reproducible fleet policy comparison without opening the dashboard."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.main import parse_cutoff, paths
from mobility.inference import forecast
from mobility.simulation import SimulationConfig, compare_policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', required=True, help='Hour with an explicit offset, e.g. 2026-04-01T08:00:00-04:00')
    parser.add_argument('--hours', type=int, default=2)
    parser.add_argument('--fleet-size', type=int, default=80)
    parser.add_argument('--fuel-price', type=float, default=3.5)
    parser.add_argument('--mpg', type=float, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, help='Optional full replay JSON output')
    args = parser.parse_args()
    data, artifacts = paths()
    cfg = SimulationConfig(start=parse_cutoff(args.as_of), hours=args.hours, fleet_size=args.fleet_size,
                           fuel_price=args.fuel_price, mpg=args.mpg, seed=args.seed)
    predictions, _, _ = forecast(data, artifacts, cfg.start, top_k=100000)
    result = compare_policies(data, cfg, predictions)
    if args.output:
        args.output.write_text(json.dumps(result))
    print(json.dumps({k: result[k] for k in ('start_time', 'request_count', 'net_uplift')}))
    for policy in ('baseline', 'optimized'):
        print(json.dumps({'policy': policy, **result[policy]['summary']}))


if __name__ == '__main__':
    main()
