"""Genetic Evolution Module for Strategy Optimization.

Manages a population of strategies (genomes) and evolves them over generations.
"""
import random
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)

@dataclass
class Genome:
    strategy: str
    params: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parents: List[str] = None

class PopulationManager:
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Genome] = []
        self.generation = 0
        self.history_file = Path("data/genetic_population.json")
        
    def load(self):
        """Load population from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.generation = data.get("generation", 0)
                    self.population = [Genome(**g) for g in data.get("population", [])]
                LOGGER.info(f"Loaded population: {len(self.population)} genomes, Gen {self.generation}")
            except Exception as e:
                LOGGER.error(f"Failed to load population: {e}")

    def save(self):
        """Save population to disk."""
        data = {
            "generation": self.generation,
            "population": [asdict(g) for g in self.population]
        }
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def seed_population(self, strategies: List[str], param_grids: Dict[str, Dict]):
        """Seed initial population with random parameters."""
        if self.population:
            return # Already seeded
            
        LOGGER.info("Seeding initial population...")
        for _ in range(self.population_size):
            strat = random.choice(strategies)
            grid = param_grids.get(strat, {})
            params = {}
            for k, v in grid.items():
                if isinstance(v, list):
                    params[k] = random.choice(v)
                else:
                    params[k] = v
            
            self.population.append(Genome(strategy=strat, params=params, generation=0))
        self.save()

    def update_fitness(self, results: List[Dict[str, Any]]):
        """Update fitness of genomes based on backtest results."""
        # Map results to genomes (simple matching by params/strategy)
        # In a real system, we'd track genome IDs. Here we fuzzy match.
        for res in results:
            strat = res.get("strategy")
            params = res.get("params")
            score = (res.get("metrics") or {}).get("dsr", 0.0)
            
            for genome in self.population:
                if genome.strategy == strat and genome.params == params:
                    genome.fitness = score
                    break
        self.save()

    def evolve(self, param_grids: Dict[str, Dict]):
        """Create next generation."""
        if not self.population:
            return

        LOGGER.info(f"Evolving Generation {self.generation} -> {self.generation + 1}")
        
        # 1. Selection (Tournament)
        parents = self._select_parents()
        
        # 2. Crossover & Mutation
        next_gen = []
        
        # Elitism: Keep top 10%
        parents.sort(key=lambda x: x.fitness, reverse=True)
        elite_count = int(self.population_size * 0.1)
        next_gen.extend(parents[:elite_count])
        
        while len(next_gen) < self.population_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            child = self._crossover(p1, p2)
            self._mutate(child, param_grids)
            
            child.generation = self.generation + 1
            next_gen.append(child)
            
        self.population = next_gen
        self.generation += 1
        self.save()

    def _select_parents(self) -> List[Genome]:
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            candidates = random.sample(self.population, k=3)
            winner = max(candidates, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def _crossover(self, p1: Genome, p2: Genome) -> Genome:
        """Uniform crossover."""
        # Only crossover if same strategy, else pick one
        if p1.strategy != p2.strategy:
            return Genome(
                strategy=p1.strategy, 
                params=p1.params.copy(), 
                parents=[f"Gen{p1.generation}_{p1.strategy}"]
            )
            
        child_params = {}
        for k in p1.params:
            child_params[k] = p1.params[k] if random.random() < 0.5 else p2.params.get(k, p1.params[k])
            
        return Genome(
            strategy=p1.strategy,
            params=child_params,
            parents=[f"Gen{p1.generation}", f"Gen{p2.generation}"]
        )

    def _mutate(self, genome: Genome, param_grids: Dict[str, Dict]):
        """Random mutation."""
        if random.random() < self.mutation_rate:
            grid = param_grids.get(genome.strategy, {})
            if not grid: return
            
            # Mutate one parameter
            param_to_change = random.choice(list(genome.params.keys()))
            if param_to_change in grid:
                options = grid[param_to_change]
                if isinstance(options, list):
                    genome.params[param_to_change] = random.choice(options)
