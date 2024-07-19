from llm_benchmark.app.configs import (
    BenchmarkConfig,
    ControllerConfig,
    TaskConfig,
    PromptConfig,
    ClientConfig,
    DatasetConfig,
    EvaluatorConfig,
    MetricConfig,
)
from llm_benchmark.app.benchmark_controller import BenchmarkController
from llm_benchmark.components.clients import AnthropicClient
from llm_benchmark.components.tasks import Task
from llm_benchmark.components.prompts import Prompt
from llm_benchmark.components.metrics import Metric, CostMetric, LatencyMetric
from llm_benchmark.components.evaluators import Evaluator
from llm_benchmark.components.datasets import Dataset
from llm_benchmark.components.results import Result

controller_config = ControllerConfig(use_streaming=True, save_path="results")
controller = BenchmarkController(controller_config)

task_config = TaskConfig(name="temp")
client_config = ClientConfig(
    name="anthropic",
    type="research",
    model_name="claude-3-5-sonnet-20240620",
    model_options={"max_tokens": 512, "top_p": 1, "temperature": 0},
)
prompt_config = PromptConfig(name="test", path="prompts/test.yaml", version=1)
dataset_config = DatasetConfig(data_path="datasets/test.csv", label_col="true_label")

metrics = [
    MetricConfig(type="cost"),
    MetricConfig(type="latency"),
    MetricConfig(type="accuracy", aspect="classification"),
]

evaluator_config = EvaluatorConfig(metrics=metrics)

config = BenchmarkConfig(
    project="test",
    task_config=task_config,
    client_config=client_config,
    prompt_config=prompt_config,
    dataset_config=dataset_config,
    evaluator_config=evaluator_config,
    controller_config=controller_config,
)

task = Task(task_config)
client = AnthropicClient(client_config)
prompt = Prompt(prompt_config)
dataset = Dataset(dataset_config)
evaluator = Evaluator(evaluator_config)

inference_results = controller.run_inference(task, dataset, client, prompt)

evaluation_results = controller.run_evaluation(evaluator, inference_results)

controller.save_results(inference_results, evaluation_results, "a", "b", config)
