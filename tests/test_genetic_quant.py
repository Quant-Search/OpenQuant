"""
Test for Genetic Quant Strategy.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from openquant.strategies.genetic_strategy import GeneticStrategy
from openquant.optimization.genetic.genome import generate_random_genome

def test_genetic_quant():
    print("\n--- Testing Genetic Quant Strategy ---")
    
    # 1. Generate Synthetic Data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    price = 100 + np.cumsum(np.random.normal(0, 1, n))
    volume = np.random.lognormal(10, 1, n)
    
    df = pd.DataFrame({
        "Open": price,
        "High": price + 1,
        "Low": price - 1,
        "Close": price + np.random.normal(0, 0.1, n),
        "Volume": volume
    }, index=dates)
    
    # 2. Generate Random Genome
    genome = generate_random_genome()
    print(f"Generated Genome: {genome}")
    
    # 3. Run Strategy
    strat = GeneticStrategy(genome=genome)
    signals = strat.generate_signals(df)
    
    print(f"Signals Generated: {len(signals)}")
    print(f"Signal Counts: {signals.value_counts().to_dict()}")
    
    assert len(signals) == n
    assert not signals.isnull().all()
    
    # 4. Mutate and Re-run
    print("Mutating Genome...")
    genome.mutate()
    print(f"Mutated Genome: {genome}")
    
    signals_mut = strat.generate_signals(df)
    print(f"Mutated Signal Counts: {signals_mut.value_counts().to_dict()}")
    
    print("✅ Genetic Quant Test Passed!")

if __name__ == "__main__":
    test_genetic_quant()
