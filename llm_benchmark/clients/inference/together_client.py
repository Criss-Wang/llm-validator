import logging
import os

from together import Together

from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord

logger = logging.getLogger(__name__)


class TogetherInferenceClient(LLMInferenceClient):
    def setup_client(self):
        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    async def predict_stream(self, dataset: DatasetRecord):
        messages = [
            dict(
                role=message.role,
                content=message.content,
            )
            for message in dataset.messages
        ]

        stream = self.client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            **self._model_options,
            stream=True,
        )

        for chunk in stream:
            yield dict(
                text=chunk.choices[0].delta.content,
                raw_response=chunk,
            )
