from abc import ABC, abstractmethod
from torch import Tensor

from typing import List, Dict, Any
from peint.data.utils import Protein


class Metric(ABC):
    """Abstract base class for metrics
    """
    def __init__(self, name: str):
        """Initialize the metric with a name.
        
        Args:
            name (str): The name of the metric.
        """
        self.name = name

    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        """Compute the metric value.
        
        This method should be implemented by subclasses to compute the metric based on the provided arguments.
        
        Returns:
            The computed metric value.
        """
        pass

    @abstractmethod
    def aggregate(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate multiple metric values into a single value.
        
        This method should be implemented by subclasses to define how to aggregate multiple metric values.
        
        Args:
            metrics (Dict[str, Any]): A dictionary of metric values to aggregate.
        
        Returns:
            A dictionary with aggregated metric values.
        """
        pass


class GenerativeMetric(Metric):
    """Abstract base class for metrics that evaluate the protein samples obtained from a generative model
    """
    @abstractmethod
    def compute(self, samples: List[Protein], *args, **kwargs) -> Dict[str, Any]:
        pass


class PredictiveMetric(Metric):
    """Abstract base class for metrics that evaluate the predictions made by a model
    """
    @abstractmethod
    def compute(self, predictions: Tensor, targets: Tensor, *args, **kwargs) -> Dict[str, Any]:
        pass
