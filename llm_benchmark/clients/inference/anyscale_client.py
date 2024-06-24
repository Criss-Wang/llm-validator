import logging
import os

import openai

from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord

logger = logging.getLogger(__name__)


class AnyscaleInferenceClient(LLMInferenceClient):
    def setup_client(self):
        url = self._base_url
        self.client = openai.OpenAI(
            base_url=url,
            api_key=os.getenv("ANYSCALE_API_KEY"),
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
            temperature=0,
            top_p=1,
            max_tokens=512,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            yield dict(
                text=chunk.choices[0].delta.content,
                raw_response=chunk,
            )
