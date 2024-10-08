import os
from typing import List, Dict

import openai

from llm_validation.components.clients import Client
from llm_validation.app.configs import ClientConfig


class OpenAiClient(Client):
    """
    Note that the way we use anyscale endpoints are similiar to openai so we temporarily consider openai
    """

    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.model_name = config.model_name
        self.model_options = config.model_options

    async def predict_stream(self, messages: List):
        client = openai.OpenAI()
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_options,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            yield dict(
                text=chunk.choices[0].delta.content,
                raw_response=chunk,
            )

    async def predict(self, messages: List) -> Dict:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_options,
        )
        return dict(
            text=response.choices[0].message.content,
            raw_response=response,
            usage=dict(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def sync_predict(self, messages: List) -> Dict:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_options,
        )
        return dict(
            text=response.choices[0].message.content,
            raw_response=response,
            usage=dict(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )
