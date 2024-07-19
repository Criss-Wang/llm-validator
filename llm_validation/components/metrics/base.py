from abc import ABC, abstractmethod

from llm_validation.app.configs import MetricConfig
from llm_validation.components.results import Result


class Metric(ABC):
    def __init__(self, config: MetricConfig):
        self.type = config.type
        self.stats = {}

    @abstractmethod
    def measure(self, results: Result):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_stats(self):
        pass

    @abstractmethod
    def get_scores(self):
        pass
