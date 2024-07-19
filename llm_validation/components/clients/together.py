import os
from typing import List, Dict

from together import Together

from llm_validation.components.clients import Client
from llm_validation.app.configs import ClientConfig


class TogetherClient(Client):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.api_key = os.getenv("TOGETHER_API_KEY")

    async def predict_stream(self, messages: List):
        client = Together(api_key=self.api_key)
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_options,
            stream=True,
        )

        for chunk in stream:
            if chunk.usage:
                self.input_tokens = chunk.usage.prompt_tokens
            yield dict(
                text=chunk.choices[0].delta.content,
                raw_response=chunk,
            )

    async def predict(self, messages: List) -> Dict:
        client = Together(api_key=self.api_key)
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

    def extract_usage(self, type: str = "input") -> int:
        if type == "input" and self.input_tokens:
            return self.input_tokens
