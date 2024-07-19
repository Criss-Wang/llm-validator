from llm_validation.app.configs import (
    ClientType,
    ClientConfig,
    PromptConfig,
    DatasetConfig,
    EvaluatorConfig,
    TaskConfig,
)
from llm_validation.components.clients import (
    Client,
)
from llm_validation.components.tasks import Task
from llm_validation.components.prompts import Prompt
from llm_validation.components.evaluators import Evaluator
from llm_validation.components.datasets import Dataset


def init_client(config: ClientConfig) -> Client:
    client_class = {
        ClientType.LLMGateway: LLMGatewayInferenceClient,
        ClientType.VLLM: VLLMInferenceClient,
        ClientType.ResearchLLM: ResearchInferenceClient,
    }.get(config.type)

    if client_class is None:
        raise ValueError(f"Unsupported inference client type: {config.type}")

    return client_class(**config)


def init_prompt(config: PromptConfig) -> Prompt:
    pass


def init_dataset(config: DatasetConfig) -> Dataset:
    pass


def init_evaluator(config: EvaluatorConfig) -> Evaluator:
    pass


def init_task(config: TaskConfig) -> Task:
    pass
