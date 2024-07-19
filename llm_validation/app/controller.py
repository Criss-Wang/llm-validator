import asyncio
import json

import mlflow
import pandas as pd

from llm_validation.app.configs import ValidationConfig, ControllerConfig
from llm_validation.components.clients import Client
from llm_validation.components.tasks import Task
from llm_validation.components.prompts import Prompt
from llm_validation.components.evaluators import Evaluator
from llm_validation.components.datasets import Dataset
from llm_validation.components.results import Result, InferenceResult, EvaluationResult
from llm_validation.utilities.common_utils import flatten_dict


class ValidationController:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.use_streaming = config.use_streaming
        self.parallelism = config.parallelism
        self.save_path = config.save_path

    def run_inference(
        self, task: Task, dataset: Dataset, client: Client, prompt: Prompt
    ) -> Result:
        # save prompt info
        mlflow.log_text(json.dumps(prompt.messages, indent=2), "prompt_template.json")

        # parallelly process task
        results, labels = asyncio.get_event_loop().run_until_complete(
            task.arun(client, prompt, dataset, self.use_streaming, self.parallelism)
        )
        return InferenceResult(results, labels)

    def run_evaluation(self, evaluator: Evaluator, inference_results: Result) -> Result:
        evaluation_scores = evaluator.evaluate(inference_results)
        # aggregated_metrics = evaluator.aggregate()
        aggregated_metrics = {}
        return EvaluationResult(
            evaluation_scores=evaluation_scores, aggregated_metrics=aggregated_metrics
        )

    def save_results(
        self,
        inference_results: InferenceResult,
        evaluation_results: EvaluationResult,
        experiment_name: str,
        run_name: str,
        config: ValidationConfig,
    ):
        """
        Things to save to mlflow and local:
        - metric scores
        - full stats for each request
        - list of good cases & bad cases
        -
        """
        results_table = inference_results.to_df()
        scores_table = evaluation_results.to_df()

        # save evalution results
        final_df = pd.concat([results_table, scores_table], axis=1)
        final_df.to_csv(f"results/{experiment_name}/{run_name}.csv")
        mlflow.log_artifact(f"results/{experiment_name}/{run_name}.csv")

        # save aggregated metrics
        mlflow.log_metrics(flatten_dict(evaluation_results.aggregated_metrics))
        mlflow.log_text(
            json.dumps(evaluation_results.aggregated_metrics, indent=2),
            "aggregated_metrics.json",
        )
        mlflow.log_text(json.dumps(config.dict(), indent=2), "config.json")
