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
        self.api_key = os.getenv("OPENAI_API_KEY")

    async def predict_stream(self, messages: List):
        client = openai.OpenAI(api_key=self.api_key)
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_options,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage:
                self.input_tokens = chunk.usage.prompt_tokens
            if not chunk.choices:
                continue
            yield dict(
                text=chunk.choices[0].delta.content,
                raw_response=chunk,
            )

    async def predict(self, messages: List) -> Dict:
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_options,
        )

        return dict(
            text=response.choices[0].message.content,
            raw_response=response,
            usage=dict(input_tokens=19, output_tokens=23),
        )

    def sync_predict(self, messages: List) -> Dict:
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_options,
        )

        return dict(
            text=response.choices[0].message.content,
            raw_response=response,
            usage=dict(input_tokens=19, output_tokens=23),
        )

    def extract_usage(self, type: str = "input") -> int:
        if type == "input" and self.input_tokens:
            return self.input_tokens
