"""Launch the local API and built dashboard from any working directory."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--demo', action='store_true', help='Use the small January 2025 smoke checkpoint')
    args = parser.parse_args()
    if args.demo:
        os.environ['NYC_DATA_DIR'] = str(ROOT / 'data/quick')
        os.environ['NYC_ARTIFACTS_DIR'] = str(ROOT / 'artifacts')
    if not (ROOT / 'frontend/dist/index.html').is_file():
        raise SystemExit('Build the dashboard first: npm --prefix frontend run build')
    import uvicorn
    uvicorn.run('backend.app.main:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
