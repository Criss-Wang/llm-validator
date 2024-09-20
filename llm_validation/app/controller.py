import asyncio
import os
import os
import asyncio
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
        self.extraction_config = config.extraction_config
        self.use_streaming = config.use_streaming
        self.parallelism = config.parallelism
        self.save_inference = config.save_inference
        self.save_path = config.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run_inference(
        self, task: Task, dataset: Dataset, client: Client, prompt: Prompt
    ) -> Result:
        # parallelly process task
        results, labels = asyncio.get_event_loop().run_until_complete(
            task.arun(
                client,
                prompt,
                dataset,
                self.use_streaming,
                self.parallelism,
            )
        )

        # save inference results if requested
        if self.save_inference:
            inference_df = pd.DataFrame(results)
            inference_df["label"] = labels
            inference_df.to_csv(
                self.save_path + f"/{client.name}-inference-results.csv"
            )
        return InferenceResult(
            dataset, results, labels, extraction_config=self.extraction_config
        )

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
        verbose: int,
    ):
        results_table = inference_results.to_df()
        scores_table = evaluation_results.to_df()
        project = config.project
        task = config.task_config.name
        experiment = experiment_name

        # save evalution results
        final_df = pd.concat([results_table, scores_table], axis=1)
        file_path = f"results/{project}/{task}/{experiment}/{run_name}.csv"
        folder_path = os.path.dirname(file_path)

        # create directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        final_df.to_csv(file_path)

        wandb.save(file_path)

        # Log aggregated metrics
        wandb.log(flatten_dict(evaluation_results.aggregated_metrics))

        # Save and log aggregated metrics as a JSON file
        wandb_save_file(
            evaluation_results.aggregated_metrics,
            folder_path,
            "aggregated_metrics",
        )
        # Save and log configuration as a JSON file
        wandb_save_file(config.dict(), folder_path, "config")

        if verbose >= 1:
            print("=====Selected Responses=====")
            correctness_column_name = next(
                (
                    col
                    for col in final_df.columns
                    if col.endswith("_correctness") or col.endswith("_quality")
                ),
                None,
            )
            if correctness_column_name:
                incorrect_classifications = final_df[
                    ~final_df[correctness_column_name].astype(bool)
                ]
                print(
                    incorrect_classifications[
                        ["response", "extracted_response", "label"]
                    ]
                )

                if verbose >= 2:
                    for i, row in incorrect_classifications.iterrows():
                        print(f"=====Request {i}=====")
                        print(f"Messages: {row['messages'][-1]['content']}")
                        print(f"Response: {row['response']}")
                        print(f"Extracted Response: {row['extracted_response']}")
                        print(f"Label: {row['label']}", "\n")
            else:
                print("No correctness or quality column found in the results.")
