import logging
import os

import anthropic

from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord

logger = logging.getLogger(__name__)


class AnthropicInferenceClient(LLMInferenceClient):
    def setup_client(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def predict_stream(self, dataset: DatasetRecord):
        messages = [
            dict(
                role=message.role,
                content=message.content,
            )
            for message in dataset.messages
        ]

        stream = self.client.messages.create(
            model=self._model_id,
            system=messages[0]["content"],
            messages=messages[1:],
            stream=True,
            **self._model_options,
        )

        for chunk in stream:
            if chunk.type == "content_block_delta":
                yield dict(
                    text=chunk.delta.text,
                    raw_response=chunk,
                )
            else:
                continue
