from typing import Dict, Any, Optional
import pandas as pd
from .base import BaseStrategy
from ..optimization.genetic.genome import Genome, generate_random_genome

class GeneticStrategy(BaseStrategy):
    """A strategy that uses a Genetic Genome to generate signals."""
    
    def __init__(self, genome: Optional[Genome] = None):
        self.genome = genome if genome else generate_random_genome()
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series()
            
        # Evaluate genome to get entry signals (1 or 0)
        entries = self.genome.evaluate(df)
        
        # Convert to -1, 0, 1
        # Currently Genome only produces 1 (True). 
        # We can treat 0 as "no signal" (hold).
        # If we want shorts, we need a short_condition in Genome.
        
        # For now, let's assume Long Only for Genetic Strategies
        signals = entries.replace({True: 1, False: 0})
        
        return signals

    def __str__(self):
        return f"GeneticStrategy({self.genome})"
