from llm_validation.app.configs import MetricConfig
from llm_validation.components.metrics import Metric
from llm_validation.components.metrics.accuracy import (
    ClassificationAccuracy,
    CodeGenerationAccuracy,
    CodeExplanationAccuracy,
    QuestionAnsweringAccuracy,
    SummarizationAccuracy,
    FunctionCallingAccuracy,
)
from llm_validation.components.metrics.cost import CostMetric
from llm_validation.components.metrics.latency import LatencyMetric
from llm_validation.components.metrics.security import SecurityMetric
from llm_validation.components.metrics.stability import StabilityMetric

ACCURACY_METRIC_MAPPING = {
    "classification-all": ClassificationAccuracy,
    "code-generation": CodeGenerationAccuracy,
    "code-explanation": CodeExplanationAccuracy,
    "qa": QuestionAnsweringAccuracy,
    "summarization": SummarizationAccuracy,
    "function-calling": FunctionCallingAccuracy,
}


def init_metric(config: MetricConfig) -> Metric:
    if config.type == "cost":
        return CostMetric(config)
    elif config.type == "latency":
        return LatencyMetric(config)
    elif config.type == "accuracy":
        return ACCURACY_METRIC_MAPPING[config.aspect](config)
    elif config.type == "security":
        return SecurityMetric(config)
    elif config.type == "stability":
        return StabilityMetric(config)
    else:
        raise ValueError(f"metric type undefined: {config.type}")
