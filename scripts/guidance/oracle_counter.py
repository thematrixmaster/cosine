"""Wrapper class to count oracle calls across guidance scripts."""

from typing import List, Union


class OracleCallCounter:
    """Transparent wrapper that counts the number of sequences scored by an oracle."""

    def __init__(self, oracle):
        """Initialize the counter wrapper.

        Args:
            oracle: Any oracle instance with predict/predict_batch methods
        """
        self.oracle = oracle
        self.total_sequences = 0
        self.__class__ = type('Wrapped' + oracle.__class__.__name__, (OracleCallCounter, oracle.__class__), {})

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def compute_fitness_gradient(self, sequence: str, increment=True):
        if increment:
            self.total_sequences += 1
        return self.oracle.compute_fitness_gradient(sequence)

    def predict(self, sequence: str, increment: bool = True):
        """Score a single sequence and increment counter.

        Args:
            sequence: Protein sequence string

        Returns:
            Oracle prediction (type depends on oracle implementation)
        """
        if increment:
            self.total_sequences += 1
        return self.oracle.predict(sequence)

    def predict_batch(self, sequences: List[str], increment: bool = True):
        """Score a batch of sequences and increment counter.

        Args:
            sequences: List of protein sequence strings

        Returns:
            Oracle predictions (type depends on oracle implementation)
        """
        if increment:
            self.total_sequences += len(sequences)
        return self.oracle.predict_batch(sequences)

    def get_call_count(self) -> int:
        """Return the total number of sequences scored.

        Returns:
            Total number of oracle calls (sequences scored)
        """
        return self.total_sequences

    def reset_call_count(self):
        """Reset the oracle call counter to zero."""
        self.total_sequences = 0

    def __call__(self, sequences: Union[str, List[str]], increment: bool = True):
        """Handle callable interface for oracles used as functions."""
        if isinstance(sequences, str):
            return self.predict(sequences, increment=increment)
        return self.predict_batch(sequences, increment=increment)