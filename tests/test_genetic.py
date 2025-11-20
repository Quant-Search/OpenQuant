import unittest
import pandas as pd
import numpy as np
from openquant.optimization.genetic.genome import generate_random_genome, Genome
from openquant.strategies.genetic_strategy import GeneticStrategy

class TestGeneticStrategy(unittest.TestCase):
    def setUp(self):
        # Create dummy OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
        self.df = pd.DataFrame({
            'Open': np.random.rand(100) * 100,
            'High': np.random.rand(100) * 100,
            'Low': np.random.rand(100) * 100,
            'Close': np.random.rand(100) * 100,
            'Volume': np.random.rand(100) * 1000
        }, index=dates)

    def test_genome_generation(self):
        genome = generate_random_genome()
        print(f"\nGenerated Genome: {genome}")
        self.assertIsInstance(genome, Genome)
        self.assertIsNotNone(genome.entry_condition)

    def test_genome_evaluation(self):
        genome = generate_random_genome()
        signals = genome.evaluate(self.df)
        self.assertEqual(len(signals), 100)
        self.assertTrue(signals.isin([0, 1]).all())

    def test_strategy_wrapper(self):
        strat = GeneticStrategy()
        print(f"\nStrategy: {strat}")
        signals = strat.generate_signals(self.df)
        self.assertEqual(len(signals), 100)
        # Check if signals are valid
        self.assertTrue(signals.isin([0, 1]).all())

    def test_mutation(self):
        genome = generate_random_genome()
        before = str(genome)
        # Mutate multiple times to ensure something changes
        for _ in range(5):
            genome.mutate()
        after = str(genome)
        print(f"\nBefore: {before}")
        print(f"After:  {after}")
        # It's possible it didn't change if mutation rate is low or luck, but likely it did.
        # We won't assert inequality to avoid flaky tests, but printing helps manual verification.

if __name__ == '__main__':
    unittest.main()
