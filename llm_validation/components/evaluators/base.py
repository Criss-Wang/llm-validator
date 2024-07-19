from typing import List, Dict

from llm_validation.app.configs import EvaluatorConfig
from llm_validation.components.results import Result
from llm_validation.components.factories.metric_factory import init_metric


class Evaluator:
    """
    An evaluator should be able to achieve the following things:
    - compare inference results with ground truth labels
    - aggregate stats
    - compute metrics
        - cost metrics
        - latency metrics
        - accuracy metrics
        - stability metrics
        - security metrics
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.metrics = self.load_metrics(config.metrics)

    def evaluate(self, results: Result) -> Dict:
        """
        We split evaluation process from aggregation process because of
        potential async calls from llm-judge in the `measure` function in the future
        """
        evaluation_scores = {}
        for idx in range(len(self.metrics)):
            self.metrics[idx].measure(results)

        for metric in self.metrics:
            evaluation_scores[metric.get_name()] = metric.get_scores()

        return evaluation_scores

    def aggregate(self) -> Dict:
        aggregated_metrics = {}
        for idx in range(len(self.metrics)):
            self.metrics[idx].aggregate()

        for metric in self.metrics:
            aggregated_metrics[metric.get_name()] = metric.get_stats()
        self.generate_report()
        return aggregated_metrics

    def load_metrics(self, metrics_configs: List):
        metrics = [init_metric(metrics_config) for metrics_config in metrics_configs]
        return metrics

    def generate_report(self):
        for metric in self.metrics:
            print(f"-------- {metric.get_name()} ----------")
            print(
                "\n".join(
                    f"{metric}: {val}" for metric, val in metric.get_stats().items()
                )
            )
