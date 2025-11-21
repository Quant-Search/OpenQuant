#!/usr/bin/env python3
"""
Run Genetic Optimization for a specific symbol.
Usage:
    python scripts/run_genetic_optimization.py --symbol BTC/USDT --pop-size 50 --generations 10
"""
import argparse
import logging
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from openquant.utils.logging import get_logger
from openquant.optimization.genetic.population import Population
from openquant.data.ccxt_source import fetch_ohlcv

LOGGER = get_logger("genetic_opt")

def save_best_genome(genome, filepath: Path):
    """Save the best genome to a JSON file."""
    data = {
        "genome_str": str(genome),
        "fitness": genome.fitness
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    LOGGER.info(f"Saved best genome to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Run Genetic Optimization")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to optimize (e.g., BTC/USDT)")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g., 1h, 1d)")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--output", type=str, default="data/genetic_best.json", help="Output file for best genome")
    
    args = parser.parse_args()
    
    LOGGER.info(f"Starting Genetic Optimization for {args.symbol} {args.timeframe}")
    
    # 1. Fetch Data
    LOGGER.info("Fetching data...")
    if args.exchange.lower() == "mt5":
        try:
            from openquant.data.mt5_source import fetch_ohlcv as mt5_fetch
            # MT5 fetch usually takes (symbol, timeframe, since, limit)
            # We'll use a default 'since' or just limit if supported.
            # mt5_source.fetch_ohlcv signature: (symbol, timeframe, since=None, limit=None)
            df = mt5_fetch(args.symbol, timeframe=args.timeframe, limit=1000)
        except ImportError:
            LOGGER.error("MT5 module not found. Please install MetaTrader5 package.")
            return
        except Exception as e:
            LOGGER.error(f"MT5 fetch failed: {e}")
            return
    else:
        df = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, limit=1000)
    
    if df.empty:
        LOGGER.error("No data found. Exiting.")
        return
        
    LOGGER.info(f"Loaded {len(df)} bars.")
    
    # 2. Initialize Population
    pop = Population(size=args.pop_size)
    LOGGER.info(f"Initialized population of size {args.pop_size}")
    
    # 3. Evolution Loop
    for gen in range(args.generations):
        pop.evolve(df)
        best = pop.best_genome
        avg_fitness = sum(g.fitness for g in pop.genomes) / len(pop.genomes)
        
        LOGGER.info(f"Gen {gen+1}/{args.generations} | Best Fitness: {best.fitness:.4f} | Avg Fitness: {avg_fitness:.4f}")
        
        # Save intermediate result
        save_best_genome(best, Path(args.output))
        
    LOGGER.info("Optimization complete.")
    LOGGER.info(f"Best Genome Genes: {str(pop.best_genome)}")

if __name__ == "__main__":
    main()
