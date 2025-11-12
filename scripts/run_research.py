"""CLI wrapper to run research from a config file."""
from __future__ import annotations
from pathlib import Path
import sys

# Ensure repository root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openquant.research.runner import run_from_config


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/aapl_sma.yaml"
    p = run_from_config(cfg)
    print(p)


if __name__ == "__main__":
    main()

