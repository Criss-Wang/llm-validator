import os
from typing import List, Dict

import anthropic

from llm_validation.components.clients import Client
from llm_validation.app.configs import ClientConfig


class AnthropicClient(Client):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    async def predict_stream(self, messages: List):
        client = anthropic.Anthropic(api_key=self.api_key)
        stream = client.messages.create(
            model=self.model_name,
            system=messages[0]["content"],
            messages=messages[1:],
            stream=True,
            **self.model_options,
        )

        for chunk in stream:
            if chunk.type == "message_start":
                self.input_tokens = chunk.message.usage.input_tokens
            elif chunk.type == "content_block_delta":
                yield dict(
                    text=chunk.delta.text,
                    raw_response=chunk,
                )
            else:
                continue

    async def predict(self, messages: List) -> Dict:
        """
        Message(
            id='msg_01MjXH1yu3f6GAjkjq9V1J8F',
            content=[TextBlock(text='text_content', type='text')],
            model='claude-3-5-sonnet-20240620',
            role='assistant',
            stop_reason='end_turn',
            stop_sequence=None,
            type='message',
            usage=Usage(input_tokens=19, output_tokens=23)
        )
        """
        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model_name,
            system=messages[0]["content"],
            messages=messages[1:],
            **self.model_options,
        )
        return dict(
            text=response.content[0].text,
            raw_response=response,
            usage=dict(response.usage),
        )

    def sync_predict(self, messages: List) -> Dict:
        """
        Message(
            id='msg_01MjXH1yu3f6GAjkjq9V1J8F',
            content=[TextBlock(text='text_content', type='text')],
            model='claude-3-5-sonnet-20240620',
            role='assistant',
            stop_reason='end_turn',
            stop_sequence=None,
            type='message',
            usage=Usage(input_tokens=19, output_tokens=23)
        )
        """
        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model_name,
            system=messages[0]["content"],
            messages=messages[1:],
            **self.model_options,
        )
        return dict(
            text=response.content[0].text,
            raw_response=response,
            usage=dict(response.usage),
        )

    def extract_usage(self, type: str = "input") -> int:
        if type == "input" and self.input_tokens:
            return self.input_tokens

    def extract_usage(self, type: str = "input") -> int:
        if type == "input" and self.input_tokens:
            return self.input_tokens
