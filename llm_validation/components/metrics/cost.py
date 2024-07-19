from typing import Dict, List

from .base import Metric
from llm_validation.components.results import Result


class CostMetric(Metric):
    def measure(self, results: Result):
        self.input_usages = [st["input_usage"] for st in results.token_statistics]
        self.output_usages = [st["output_usage"] for st in results.token_statistics]
        self.size = len(results)

    def aggregate(self):
        self.aggregate_input_tokens()
        self.aggregate_output_tokens()

    def aggregate_input_tokens(self):
        average_input_tokens = 0
        min_input_tokens = 1e9
        max_input_tokens = 0

        for input_tokens in self.input_usages:
            average_input_tokens += input_tokens
            min_input_tokens = min(min_input_tokens, input_tokens)
            max_input_tokens = max(max_input_tokens, input_tokens)

        average_input_tokens /= self.size
        self.stats.update(
            {
                "average_input_tokens": average_input_tokens,
                "min_input_tokens": min_input_tokens,
                "max_input_tokens": max_input_tokens,
            }
        )

    def aggregate_output_tokens(self):
        average_output_tokens = 0
        min_output_tokens = 1e9
        max_output_tokens = 0

        for output_tokens in self.output_usages:
            average_output_tokens += output_tokens
            min_output_tokens = min(min_output_tokens, output_tokens)
            max_output_tokens = max(max_output_tokens, output_tokens)

        average_output_tokens /= self.size
        self.stats.update(
            {
                "average_output_tokens": average_output_tokens,
                "min_output_tokens": min_output_tokens,
                "max_output_tokens": max_output_tokens,
            }
        )

    def get_name(self):
        return "Cost"

    def get_stats(self):
        return self.stats

    def get_scores(self):
        return {
            "input_usages": self.input_usages,
            "output_usages": self.output_usages,
        }
