import asyncio
import json
import logging
from typing import List

import click
import mlflow
import numpy as np

from llm_benchmark.clients.inference_factories import create_inference_client
from llm_benchmark.data.etl import load_dataset_from_df, initialize_aspects, load_prompt
from llm_benchmark.task import Task
from llm_benchmark.utilities import initialize_logging, load_config

np.random.seed(42)

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(uri="http://localhost:5000")


def run_benchmark(config_path: str):
    config = load_config(path=config_path)

    client = create_inference_client(config.client)
    task = Task(
        prompt=config.prompt,
        records=(
            load_dataset_from_df(config.dataset_path, config.prompt, config.label_col)
        ),
        aspects=initialize_aspects(config.aspects),
        parallelism=config.parallelism,
    )

    run_name = f"{config.model}-{config.prompt.tenant}-{config.prompt.name}-v{config.prompt.version}"
    experiment_name = f"{config.project}-{config.task}"
    if curr_experiment := mlflow.get_experiment_by_name(experiment_name):
        runs = mlflow.search_runs([curr_experiment.experiment_id])
        if len(runs) > 0:
            old_runs_ids = runs[runs["tags.mlflow.runName"] == run_name]["run_id"]
            for id in old_runs_ids:
                mlflow.delete_run(id)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=run_name,
        tags={
            "project": config.project,
            "task": config.task,
            "model": config.model,
            "prompt_name": config.prompt.name,
            "tenant": config.prompt.tenant,
            "client": config.client.type,
        },
    ):
        asyncio.get_event_loop().run_until_complete(task.process_records(client))

        metrics = task.get_aggregated_metrics()
        result_table = task.get_aggregated_results()
        mlflow.log_text(
            json.dumps(load_prompt(config.prompt), indent=2), "prompt_used.json"
        )

        table_name = f"{config.prompt.tenant}-{config.prompt.name}-v{config.prompt.version}-{config.model.split('/')[-1]}_result_table"
        result_table.to_csv(f"results/{table_name}.csv")
        mlflow.log_artifact(f"results/{table_name}.csv")
        mlflow.log_table(result_table, f"{table_name}.json")

        mlflow.log_metrics(metrics)
        mlflow.log_text(json.dumps(metrics, indent=2), "aggregated_metrics.json")
        mlflow.log_text(json.dumps(config.dict(), indent=2), "config.json")
        mlflow.log_text(json.dumps(task.results_to_dict(), indent=2), "results.json")
        mlflow.log_text(json.dumps(task.dataset_to_dict(), indent=2), "dataset.json")


@click.group()
def cli():
    initialize_logging()


@cli.command()
@click.option("--config-path", "configs", type=str, multiple=True, required=True)
def run(configs: List[str]):
    for config_path in configs:
        click.echo("")
        try:
            click.echo(f"Running benchmark with config: {config_path}")
            run_benchmark(config_path)
        except Exception:
            logger.exception(f"Failed running benchmark with config: {config_path}")


if __name__ == "__main__":
    cli()
