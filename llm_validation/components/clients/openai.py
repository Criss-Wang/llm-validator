import os
from typing import List, Dict

import openai

from llm_validation.components.clients import Client
from llm_validation.app.configs import ClientConfig


class OpenAiClient(Client):
    """
    Note that anyscale endpoints are similiar to
    """

    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE")
        self.model_version = os.getenv("OPENAI_API_VERSION")
        self.model_id = "GPT4"
        self.model_options = config.model_options

    async def predict_stream(self, messages: List):
        client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.model_version,
            azure_endpoint=self.base_url,
        )
        stream = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True,
            **self.model_options,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            yield dict(
                text=chunk.choices[0].delta.content,
                raw_response=chunk,
            )

    async def predict(self, messages: List) -> Dict:
        client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.model_version,
            azure_endpoint=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **self.model_options,
        )
        return dict(
            text=response.choices[0].message.content,
            raw_response=response,
            usage=dict(input_tokens=19, output_tokens=23),
        )

    def sync_predict(self, messages: List) -> Dict:
        client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.model_version,
            azure_endpoint=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **self.model_options,
        )
        return dict(
            text=response.choices[0].message.content,
            raw_response=response,
            usage=dict(input_tokens=19, output_tokens=23),
        )
