import json
import logging
import aiohttp

from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord

logger = logging.getLogger(__name__)


class SelfHostedInferenceClient(LLMInferenceClient):
    def setup_client(self):
        # load any credentials if necessary
        self.client = None

    async def predict_stream(self, dataset: DatasetRecord):
        messages = [
            dict(
                role=message.role,
                content=message.content,
            )
            for message in dataset.messages
        ]

        system_prompt, user_prompt = messages
        payload = {
            "input": {
                **self._model_options,
                "prompt": user_prompt["content"],
                "system_prompt": system_prompt["content"],
            },
            "stream": True,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{self._base_url}",
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
            ) as response:
                async for chunk in response.content:
                    chunk = chunk.decode("utf-8").lstrip("data: ").strip()
                    if chunk != "[DONE]" and chunk != "":
                        data = json.loads(chunk)

                        if not data.get("choices"):
                            logger.warning(chunk)
                            continue

                        if "content" not in data["choices"][0]["delta"]:
                            continue

                        yield dict(
                            text=data["choices"][0]["delta"]["content"],
                            raw_response=chunk,
                        )
