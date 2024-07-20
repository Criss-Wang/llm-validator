import asyncio
import os
import json
import logging

import pandas as pd
import wandb

from llm_validation.app.configs import ValidationConfig, ControllerConfig
from llm_validation.components.clients import Client
from llm_validation.components.tasks import Task
from llm_validation.components.prompts import Prompt
from llm_validation.components.evaluators import Evaluator
from llm_validation.components.datasets import Dataset
from llm_validation.components.results import Result, InferenceResult, EvaluationResult
from llm_validation.utilities.common_utils import flatten_dict, wandb_save_file

logger = logging.getLogger(__name__)


class ValidationController:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.use_streaming = config.use_streaming
        self.parallelism = config.parallelism
        self.save_path = config.save_path

    def run_inference(
        self, task: Task, dataset: Dataset, client: Client, prompt: Prompt
    ) -> Result:

        # parallelly process task
        results, labels = asyncio.get_event_loop().run_until_complete(
            task.arun(client, prompt, dataset, self.use_streaming, self.parallelism)
        )
        return InferenceResult(results, labels)

    def run_evaluation(self, evaluator: Evaluator, inference_results: Result) -> Result:
        evaluation_scores = evaluator.evaluate(inference_results)

        # ensure results saving are not blocked by aggregation logic errors
        try:
            aggregated_metrics = evaluator.aggregate()
        except Exception as e:
            logger.error(e)
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
        results_table = inference_results.to_df()
        scores_table = evaluation_results.to_df()

        # save evalution results
        final_df = pd.concat([results_table, scores_table], axis=1)

        results_path = f"results/{experiment_name}/"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        final_df.to_csv(f"{results_path}/{run_name}.csv")
        wandb.save(f"{results_path}/{run_name}.csv")

        # Log aggregated metrics
        wandb.log(flatten_dict(evaluation_results.aggregated_metrics))

        # Save and log aggregated metrics as a JSON file
        wandb_save_file(
            evaluation_results.aggregated_metrics,
            results_path,
            "aggregated_metrics",
        )
        # Save and log configuration as a JSON file
        wandb_save_file(config.dict(), results_path, "config")
