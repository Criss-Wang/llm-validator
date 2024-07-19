import time
import asyncio
from typing import List, Tuple

from tqdm.asyncio import tqdm_asyncio

from llm_validation.app.configs import TaskConfig
from llm_validation.components.clients import Client
from llm_validation.components.prompts import Prompt
from llm_validation.components.datasets import Dataset
from llm_validation.components.results import Result


class Task:
    def __init__(self, config: TaskConfig):
        self.name = config.name

    async def arun(
        self,
        client: Client,
        prompt: Prompt,
        dataset: Dataset,
        use_streaming: bool = False,
        parallelism: int = 4,
    ) -> Tuple:
        tasks = []
        semaphore = asyncio.Semaphore(parallelism)

        transformed_dataset = dataset.adopt_prompt(prompt)

        for messages in transformed_dataset:
            tasks.append(
                asyncio.create_task(
                    self._make_request(client, messages, semaphore, use_streaming)
                )
            )

        results = await tqdm_asyncio.gather(*tasks, desc="Calling LLM")
        labels = transformed_dataset.get_labels()
        return results, labels

    async def _make_request(
        self,
        client: Client,
        messages: List,
        semaphore: asyncio.Semaphore,
        use_streaming: bool,
    ) -> List:
        async with semaphore:
            time_to_first_token = None
            tokens = []

            # start of inference task
            start_time = time.time()

            # TODO: further refactor this code
            if use_streaming:
                async for token in client.predict_stream(messages):
                    if time_to_first_token is None:
                        time_to_first_token = time.time() - start_time
                    tokens.append(token)
                end_time = time.time()
                total_time = end_time - start_time
                input_usage = 0
                # handle input_usage that is given at the start of streaming
                if tokens[0]["text"] == "<Claude_Start>":
                    input_usage = client.extract_usage("input")
                    tokens = tokens[1:]

                response = "".join(
                    token["text"]
                    for token in tokens
                    if token is not None and token["text"] is not None
                )
                token_statistics = {
                    "input_usage": input_usage,
                    "output_usage": len(tokens),
                }
            else:
                results = await client.predict(messages)
                end_time = time.time()
                total_time = end_time - start_time
                response = results["text"]
                token_statistics = results["usage"]

            # end of inference task
            success = True
            time_statistics = {
                "total_time": total_time,
                "start_time": start_time,
                "end_time": end_time,
                "time_to_first_token": time_to_first_token,
            }

            return dict(
                success=success,
                messages=messages,
                response=response,
                token_statistics=token_statistics,
                time_statistics=time_statistics,
            )
