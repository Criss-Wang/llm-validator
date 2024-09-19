from abc import ABC, abstractmethod
from collections import defaultdict

from llm_validation.app.configs import MetricConfig
from llm_validation.components.results import Result


class Metric(ABC):
    def __init__(self, config: MetricConfig):
        self.type = config.type
        self.scores = defaultdict(list)
        self.stats = {}
        self.metric_keys = []

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
