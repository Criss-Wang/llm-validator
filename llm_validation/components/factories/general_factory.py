from llm_validation.app.configs import (
    PromptConfig,
    DatasetConfig,
    EvaluatorConfig,
    TaskConfig,
)

from llm_validation.components.tasks import Task
from llm_validation.components.prompts import Prompt
from llm_validation.components.evaluators import Evaluator
from llm_validation.components.datasets import Dataset


def init_prompt(config: PromptConfig) -> Prompt:
    return Prompt(config)


def init_dataset(config: DatasetConfig) -> Dataset:
    return Dataset(config)


def init_evaluator(config: EvaluatorConfig) -> Evaluator:
    return Evaluator(config)


def init_task(config: TaskConfig) -> Task:
    return Task(config)
