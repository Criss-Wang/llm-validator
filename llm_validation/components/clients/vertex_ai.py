import os
import time
from typing import List, Dict

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, GenerationConfig

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
        project = os.getenv("VERTEX_AI_GCP_PROJECT")
        location = os.getenv("VERTEX_AI_GCP_LOCATION")
        vertexai.init(project=project, location=location)

    async def predict_stream(self, messages: List):
        model = GenerativeModel(
            model_name=self.model_name,
            generation_config=self.config,
            safety_settings=self.safety_settings,
            system_instruction=messages[0]["content"],
        )
        try:
            stream = model.generate_content([messages[1]["content"]], stream=True)
        except Exception as e:
            # vertex_ai has strict credit usage limit per hour/ per min
            print(e)
            time.sleep(60)
            stream = model.generate_content([messages[1]["content"]], stream=True)

        for chunk in stream:
            try:
                if chunk.usage_metadata.prompt_token_count > 0:
                    self.input_tokens = chunk.usage_metadata.prompt_token_count
                yield dict(text=chunk.text, raw_response=chunk)
            except Exception as e:
                print(e)
                print(
                    "*** ValueError: Response has no candidates (and thus no text).",
                    "The response is likely blocked by the safety filters.",
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
        except Exception as e:
            # vertex_ai has strict credit usage limit per hour/ per min
            print(e)
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

    def extract_usage(self, type: str = "input") -> int:
        if type == "input" and self.input_tokens:
            return self.input_tokens
