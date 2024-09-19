import logging
from typing import Dict, List

from ..base import Metric
from llm_validation.components.results import Result

logger = logging.getLogger(__name__)


class LatencyMetric(Metric):
    def measure(self, results: Result):
        self.total_times = [st["total_time"] for st in results.time_statistics]
        self.time_to_first_tokens = [
            st["time_to_first_token"] for st in results.time_statistics
        ]
        self.output_tokens = [st["output_tokens"] for st in results.token_statistics]
        self.size = len(results)

    def aggregate(self):
        self.aggregate_time_to_first_token()
        self.aggregate_time_per_request()
        self.aggregate_token_per_second()

    def aggregate_time_to_first_token(self):
        if not self.time_to_first_tokens[0]:
            logger.info("No TTFT recorded, requests are non-streaming")
            return

        average_time_to_first_token = 0
        max_time_to_first_token = 0
        min_time_to_first_token = 1e9

        for time_to_first_token in self.time_to_first_tokens:
            average_time_to_first_token += time_to_first_token
            max_time_to_first_token = max(max_time_to_first_token, time_to_first_token)
            min_time_to_first_token = min(min_time_to_first_token, time_to_first_token)

        average_time_to_first_token /= self.size
        self.stats.update(
            {
                "average_time_to_first_token": average_time_to_first_token,
                "max_time_to_first_token": max_time_to_first_token,
                "min_time_to_first_token": min_time_to_first_token,
            }
        )

    def aggregate_token_per_second(self):
        average_tokens_per_second = 0
        max_tokens_per_second = 0
        min_tokens_per_second = 1e9
        for total_time, output_tokens in zip(self.total_times, self.output_tokens):
            tokens_per_second = output_tokens / total_time
            average_tokens_per_second += tokens_per_second
            max_tokens_per_second = max(max_tokens_per_second, tokens_per_second)
            min_tokens_per_second = min(min_tokens_per_second, tokens_per_second)

        average_tokens_per_second /= self.size
        self.stats.update(
            {
                "average_tokens_per_second": average_tokens_per_second,
                "max_tokens_per_second": max_tokens_per_second,
                "min_tokens_per_second": min_tokens_per_second,
            }
        )

    def aggregate_time_per_request(self):
        self.stats["average_time_per_request"] = sum(self.total_times) / self.size

    def get_name(self):
        return "Latency"

    def get_stats(self):
        return self.stats

    def get_scores(self):
        return {
            "total_time": self.total_times,
            "time_to_first_token": self.time_to_first_tokens,
        }
