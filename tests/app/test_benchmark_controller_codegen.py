from llm_validation.app.configs import (
    ValidationConfig,
    ControllerConfig,
    TaskConfig,
    PromptConfig,
    ClientConfig,
    DatasetConfig,
    EvaluatorConfig,
    MetricConfig,
)
from llm_validation.app.validation_controller import ValidationController
from llm_validation.components.clients import LocalClient
from llm_validation.components.tasks import Task
from llm_validation.components.prompts import Prompt
from llm_validation.components.evaluators import Evaluator
from llm_validation.components.datasets import Dataset

controller_config = ControllerConfig(save_path="results")
controller = ValidationController(controller_config)

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

config = ValidationConfig(
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
