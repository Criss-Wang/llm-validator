from typing import Dict

from .base import Metric
from llm_validation.components.results import Result


class SecurityMetric(Metric):
    def measure(self, results: Result):
        pass

    def aggregate(self):
        pass

    def get_name(self):
        pass

    def get_stats(self):
        pass

    def get_scores(self):
        pass
