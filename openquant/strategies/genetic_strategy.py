
import pandas as pd

from ..optimization.genetic.genome import Genome, generate_random_genome
from .base import BaseStrategy


class GeneticStrategy(BaseStrategy):
    """A strategy that uses a Genetic Genome to generate signals."""

    def __init__(self, genome: Genome | None = None) -> None:
        self.genome: Genome = genome if genome else generate_random_genome()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=int)

        entries = self.genome.evaluate(df)

        signals = entries.replace({True: 1, False: 0})

        return signals

    def __str__(self) -> str:
        return f"GeneticStrategy({self.genome})"
