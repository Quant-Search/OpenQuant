"""
Test for Population Manager.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from openquant.optimization.genetic.population import Population

def test_population():
    print("\n--- Testing Population Manager ---")
    
    # 1. Generate Synthetic Data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Create a trend that a simple strategy could catch
    price = 100 + np.cumsum(np.random.normal(0.1, 1, n)) # Upward trend
    
    df = pd.DataFrame({
        "Open": price,
        "High": price + 1,
        "Low": price - 1,
        "Close": price + np.random.normal(0, 0.1, n),
        "Volume": np.random.lognormal(10, 1, n)
    }, index=dates)
    
    # 2. Initialize Population
    pop_size = 10
    pop = Population(size=pop_size)
    print(f"Population initialized with {pop.size} genomes.")
    
    # 3. Run Evolution Loop
    generations = 3
    for i in range(generations):
        print(f"\nGeneration {i+1}...")
        pop.evolve(df)
        best = pop.best_genome
        print(f"  Best Fitness: {best.fitness:.4f}")
        print(f"  Best Genome: {best}")
        
        # Verify population size maintained
        assert len(pop.genomes) == pop_size
        
    # 4. Check improvement (not guaranteed in short run but likely)
    # At least ensure fitness is calculated
    assert pop.best_genome.fitness != -999.0
    
    print("\n✅ Population Manager Test Passed!")

if __name__ == "__main__":
    test_population()
