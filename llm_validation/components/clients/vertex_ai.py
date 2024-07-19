import os
import time
from typing import List, Dict

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai.preview.generative_models as generative_models

from llm_validation.components.clients import Client
from llm_validation.app.configs import ClientConfig


class VertexAiClient(Client):
    def __init__(self, config: ClientConfig):
        super().__init__(config)

        self.config = GenerationConfig(**config.model_options)
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

        vertexai.init(project="aisera-gemini", location="us-central1")

    async def predict_stream(self, messages: List):
        model = GenerativeModel(
            model_name=self.model_name,
            generation_config=self.config,
            safety_settings=self.safety_settings,
            system_instruction=messages[0]["content"],
        )
        try:
            stream = model.generate_content([messages[1]["content"]], stream=True)
        except:
            # vertex_ai has strict credit usage limit per hour/ per min
            time.sleep(60)
            stream = model.generate_content([messages[1]["content"]], stream=True)

        for chunk in stream:
            try:
                if not chunk.text:
                    continue
                yield dict(text=chunk.text, raw_response=chunk)
            except:
                print(
                    "*** ValueError: Response has no candidates (and thus no text). The response is likely blocked by the safety filters."
                )
                yield dict(text="", raw_response=chunk)
                break

    async def predict(self, messages: List) -> Dict:
        model = GenerativeModel(
            model_name=self.model_name,
            generation_config=self.config,
            safety_settings=self.safety_settings,
            system_instruction=messages[0]["content"],
        )
        try:
            response = model.generate_content([messages[1]["content"]])
        except:
            # vertex_ai has strict credit usage limit per hour/ per min
            time.sleep(60)
            response = model.generate_content([messages[1]["content"]])
        return dict(
            text=response.candidates[0].content.parts[0].text,
            raw_response=response,
            usage=dict(
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            ),
        )
