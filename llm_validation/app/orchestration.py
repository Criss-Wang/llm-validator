import logging
from datetime import datetime
from typing import List

import click
import mlflow

from llm_validation.app.configs import ValidationConfig
from llm_validation.app.validation_controller import ValidationController
from llm_validation.components.factories.client_factory import init_client
from llm_validation.components.factories.general_factory import (
    init_dataset,
    init_evaluator,
    init_prompt,
    init_task,
)
from llm_validation.utilities.common_utils import initialize_logging, set_random_state
from llm_validation.utilities.config_utils import load_validation_config


logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(uri="http://localhost:5000")
set_random_state()


def setup_mlflow_experiment(project: str, task_name: str, model_name: str) -> List[str]:
    run_name = f'{model_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    experiment_name = f"{project}-{task_name}"

    # replace the existing experiment-run with the same name
    if curr_experiment := mlflow.get_experiment_by_name(experiment_name):
        runs = mlflow.search_runs([curr_experiment.experiment_id])
        if len(runs) > 0:
            old_runs_ids = runs[runs["tags.mlflow.runName"] == run_name]["run_id"]
            for id in old_runs_ids:
                mlflow.delete_run(id)
    mlflow.set_experiment(experiment_name)
    return run_name, experiment_name


def run_validation(config: ValidationConfig):
    client = init_client(config.client_config)
    dataset = init_dataset(config.dataset_config)
    evaluator = init_evaluator(config.evaluator_config)
    prompt = init_prompt(config.prompt_config)
    task = init_task(config.task_config)

    validation_controller = ValidationController(config.controller_config)

    run_name, experiment_name = setup_mlflow_experiment(
        config.project, task.name, client.model_name
    )

    with mlflow.start_run(
        run_name=run_name,
        tags={
            "project": config.project,
            "task": task.name,
            "client": client.name,
            "model": client.model_name,
            "prompt": prompt.name,
        },
    ):
        inference_results = validation_controller.run_inference(
            task, dataset, client, prompt
        )
        evaluation_results = validation_controller.run_evaluation(
            evaluator, inference_results
        )
        validation_controller.save_results(
            inference_results, evaluation_results, run_name, experiment_name, config
        )


@click.group()
def cli():
    initialize_logging()


@cli.command()
@click.option("--config-path", "configs", type=str, multiple=True, required=True)
def run(configs: List[str]):
    for config_path in configs:
        click.echo("")
        try:
            click.echo(f"Running validation with config: {config_path}")
            config = load_validation_config(path=config_path)
            run_validation(config)
        except Exception:
            logger.exception(f"Failed running validation with config: {config_path}")


if __name__ == "__main__":
    cli()
