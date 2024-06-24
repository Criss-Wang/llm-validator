import logging
import os
import time

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai.preview.generative_models as generative_models

from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord

logger = logging.getLogger(__name__)

SAFETY_SETTING = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
}


class GeminiInferenceClient(LLMInferenceClient):
    def setup_client(self):
        vertexai.init(
            project=os.environ.get("VERTEX_AI_PROJECT"), location="us-central1"
        )

    async def predict_stream(self, dataset: DatasetRecord):
        messages = [
            dict(
                role=message.role,
                content=message.content,
            )
            for message in dataset.messages
        ]

        system_prompt = messages[0]["content"]
        user_message = (
            messages[1]["content"] if len(messages) > 1 else "The response is:"
        )
        model = GenerativeModel(
            self._model_id,
            system_instruction=system_prompt,
            generation_config=GenerationConfig(**self._model_options),
            safety_settings=SAFETY_SETTING,
        )
        try:
            stream = model.generate_content([user_message], stream=True)
        except:
            time.sleep(60)
            stream = model.generate_content([user_message], stream=True)

        for chunk in stream:
            try:
                if not chunk.text:
                    continue
                yield dict(
                    text=chunk.text,
                    raw_response=chunk,
                )
            except:
                print(
                    "*** ValueError: Response has no candidates (and thus no text). The response is likely blocked by the safety filters."
                )
                yield dict(
                    text="",
                    raw_response=chunk,
                )
                break
