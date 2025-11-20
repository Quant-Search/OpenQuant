"""
Population Manager for Genetic Optimization.
Manages the evolution of trading strategies.
"""
import random
import pandas as pd
import numpy as np
from typing import List, Callable, Tuple
from .genome import Genome, generate_random_genome
from openquant.strategies.genetic_strategy import GeneticStrategy
from openquant.backtest.engine import backtest_signals, sharpe

class Population:
    """Manages a population of Genomes."""
    
    def __init__(self, size: int = 50):
        self.size = size
        self.genomes: List[Genome] = [generate_random_genome() for _ in range(size)]
        self.generation = 0
        self.best_genome: Genome = None
        
    def evaluate(self, df: pd.DataFrame):
        """Evaluate all genomes on the provided data."""
        for genome in self.genomes:
            try:
                # Create strategy from genome
                strat = GeneticStrategy(genome)
                signals = strat.generate_signals(df)
                
                # Quick backtest
                # Assuming 1h data, simple fees
                res = backtest_signals(df, signals, fee_bps=1.0)
                
                # Fitness = Sharpe Ratio (or PnL)
                # Handle NaN or empty returns
                if res.returns.empty or res.returns.std() == 0:
                    genome.fitness = -999.0
                else:
                    # Annualized Sharpe (assuming hourly data -> 24*365 bars?)
                    # Actually sharpe function handles freq if passed, or defaults to daily
                    # Let's use raw Sharpe for comparison
                    s = sharpe(res.returns)
                    genome.fitness = s if not np.isnan(s) else -999.0
                    
            except Exception as e:
                genome.fitness = -999.0
                
        # Sort by fitness (descending)
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        self.best_genome = self.genomes[0]
        
    def select(self, k: int = 3) -> Genome:
        """Tournament Selection."""
        candidates = random.sample(self.genomes, k)
        return max(candidates, key=lambda g: g.fitness)
        
    def evolve(self, df: pd.DataFrame, elitism: int = 2, mutation_rate: float = 0.2):
        """Run one generation of evolution."""
        # 1. Evaluate current population
        self.evaluate(df)
        
        new_genomes = []
        
        # 2. Elitism: Keep best N
        new_genomes.extend(self.genomes[:elitism])
        
        # 3. Reproduction
        while len(new_genomes) < self.size:
            # Select parents
            p1 = self.select()
            p2 = self.select()
            
            # Crossover
            child = p1.crossover(p2)
            
            # Mutation
            if random.random() < mutation_rate:
                child.mutate()
                
            new_genomes.append(child)
            
        self.genomes = new_genomes
        self.generation += 1
        
    def get_best_strategy(self) -> GeneticStrategy:
        if self.best_genome:
            return GeneticStrategy(self.best_genome)
        return None
