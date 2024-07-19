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
from llm_benchmark.components.clients import LocalClient
from llm_benchmark.components.tasks import Task
from llm_benchmark.components.prompts import Prompt
from llm_benchmark.components.evaluators import Evaluator
from llm_benchmark.components.datasets import Dataset

controller_config = ControllerConfig(save_path="results")
controller = BenchmarkController(controller_config)

task_config = TaskConfig(name="codegen")
client_config = ClientConfig(
    name="local",
    type="research",
    model_name="llama3-70b-instruct-GPTQ-8bits",
)
prompt_config = PromptConfig(
    name="code-gen-prompt-v1", path="prompts/codegen.yaml", version=1
)
dataset_config = DatasetConfig(
    data_path="datasets/codegen_20240717.csv", label_col="true_label"
)

metrics = [
    MetricConfig(type="accuracy", aspect="codegen"),
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
client = LocalClient(client_config)
prompt = Prompt(prompt_config)
dataset = Dataset(dataset_config)
evaluator = Evaluator(evaluator_config)

inference_results = controller.run_inference(task, dataset, client, prompt)

evaluation_results = controller.run_evaluation(evaluator, inference_results)

controller.save_results(
    inference_results, evaluation_results, "codegen", "llama3", config
)
