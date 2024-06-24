import logging
import os

import openai


from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord

logger = logging.getLogger(__name__)


class MonolythInferenceClient(LLMInferenceClient):
    def setup_client(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("MONOLYTH_API_KEY"),
            base_url=self._base_url,
        )

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
        )
        for chunk in stream:
            yield dict(
                text=chunk.choices[0].delta.content,
                raw_response=chunk,
            )
