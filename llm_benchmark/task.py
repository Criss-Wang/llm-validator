import asyncio
import time
from typing import Any, Dict, List

import tqdm
import pandas as pd

from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from llm_benchmark.aspects.aspect import Aspect
from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord, LLMResultRecord, PromptConfig


class Task(BaseModel):
    prompt: PromptConfig
    records: List[DatasetRecord]
    aspects: List[Aspect]
    results: List[LLMResultRecord] = []
    _inference_duration: float = 0
    parallelism: int = 12

    async def process_records(self, client: LLMInferenceClient) -> None:
        tasks = []
        semaphore = asyncio.Semaphore(self.parallelism)

        start_time = time.time()
        for record in self.records:
            tasks.append(
                asyncio.create_task(
                    self._make_request(
                        client=client, record=record, semaphore=semaphore
                    )
                )
            )

        results = await tqdm_asyncio.gather(*tasks, desc="Calling LLM")
        self._inference_duration = time.time() - start_time
        for result in tqdm.tqdm(results, desc="Extracting metrics"):
            for aspect in self.aspects:
                metrics = aspect.process_record(result)

                if metrics:
                    result.log_metrics(metrics)

            self.results.append(result)

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        metrics = self._extract_system_metrics()

        for aspect in self.aspects:
            metrics.update(aspect.get_aggregated_metrics())

        return metrics

    def get_aggregated_results(self) -> pd.DataFrame:
        result_df = []
        for result in self.results:
            inputs = result.request["inputs"]
            messages = result.request["messages"]
            d = result.dict(exclude=["request"])
            d["inputs"] = inputs
            d["messages"] = messages
            result_df.append(d)
        return pd.DataFrame(result_df)

    def results_to_dict(self) -> List[Dict[str, Any]]:
        return [result.dict() for result in self.results]

    def dataset_to_dict(self) -> List[Dict[str, Any]]:
        return [record.dict() for record in self.records]

    async def _make_request(
        self,
        client: LLMInferenceClient,
        record: DatasetRecord,
        semaphore: asyncio.Semaphore,
    ) -> LLMResultRecord:
        async with semaphore:
            time_to_first_token = None
            start_time = time.time()
            tokens = []

            async for token in client.predict_stream(dataset=record):
                if time_to_first_token is None:
                    time_to_first_token = time.time() - start_time
                tokens.append(token)

            end_time = time.time()
            total_time = end_time - start_time
            return LLMResultRecord(
                prompt=self.prompt.name,
                request=dict(
                    inputs=record.inputs,
                    messages=[message.dict() for message in record.messages],
                ),
                response="".join(
                    token["text"]
                    for token in tokens
                    if token is not None and token["text"] is not None
                ),
                expected_response=record.expected_response,
                time_to_first_token=time_to_first_token,
                total_time=total_time,
                tokens_per_second=len(tokens) / total_time,
                number_of_tokens=len(tokens),
                # tokens=tokens,
                success=True,
                start_time=start_time,
                end_time=end_time,
            )

    def _extract_system_metrics(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        average_time_to_first_token = 0
        max_time_to_first_token = 0
        min_time_to_first_token = 999
        average_tokens_per_second = 0
        max_tokens_per_second = 0
        min_tokens_per_second = 999
        average_number_of_tokens = 0
        total_output_tokens = 0
        average_time_per_request = 0

        for result in self.results:
            average_time_to_first_token += result.time_to_first_token
            max_time_to_first_token = max(
                max_time_to_first_token, result.time_to_first_token
            )
            min_time_to_first_token = min(
                min_time_to_first_token, result.time_to_first_token
            )
            average_tokens_per_second += result.tokens_per_second
            max_tokens_per_second = max(max_tokens_per_second, result.tokens_per_second)
            min_tokens_per_second = min(min_tokens_per_second, result.tokens_per_second)
            average_number_of_tokens += result.number_of_tokens
            total_output_tokens += result.number_of_tokens
            average_time_per_request += result.total_time

        average_time_to_first_token /= len(self.results)
        average_tokens_per_second /= len(self.results)
        average_number_of_tokens /= len(self.results)
        average_time_per_request /= len(self.results)

        return {
            "average_time_to_first_token": average_time_to_first_token,
            "max_time_to_first_token": max_time_to_first_token,
            "min_time_to_first_token": min_time_to_first_token,
            "average_tokens_per_second": average_tokens_per_second,
            "max_tokens_per_second": max_tokens_per_second,
            "min_tokens_per_second": min_tokens_per_second,
            "average_number_of_tokens": average_number_of_tokens,
            "total_output_tokens": total_output_tokens,
            "request_throughput": len(self.results) / self._inference_duration,
            "token_throughput": total_output_tokens / self._inference_duration,
            "inference_duration": self._inference_duration,
            "average_time_per_request": average_time_per_request,
        }
