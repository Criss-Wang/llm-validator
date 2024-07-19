from llm_validation.app.configs import MetricConfig
from llm_validation.components.metrics import (
    Metric,
    CostMetric,
    LatencyMetric,
    AccuracyMetric,
    SecurityMetric,
    StabilityMetric,
)
from llm_validation.components.metrics.accuracy import (
    ClassificationAccuracy,
    CodeGenAccuracy,
)


def init_metric(config: MetricConfig) -> Metric:
    if config.type == "cost":
        return CostMetric(config)
    elif config.type == "latency":
        return LatencyMetric(config)
    elif config.type == "accuracy":
        return init_accuracy_metric(config)
    elif config.type == "security":
        return SecurityMetric(config)
    elif config.type == "stability":
        return StabilityMetric(config)
    else:
        raise ValueError(f"metric type undefined: {config.type}")


def init_accuracy_metric(config: MetricConfig) -> AccuracyMetric:
    if config.aspect == "classification":
        return ClassificationAccuracy(config)
    elif config.aspect == "codegen":
        return CodeGenAccuracy(config)
