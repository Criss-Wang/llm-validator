import json
from typing import List, Dict

import requests

from llm_validation.components.clients import Client


class LocalClient(Client):
    async def predict_stream(self, messages: List):
        raise NotImplementedError

    async def predict(self, messages: List) -> Dict:
        url = "http://localhost:8000/v1/chat/completions"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {
            "model": "cortecs/Meta-Llama-3-70B-Instruct-GPTQ-8b",
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2048,
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response = response = response.json()
        if len(response["choices"][0]["message"]["content"]) > 2000:
            print(response["choices"][0]["message"]["content"])
        return dict(
            text=response["choices"][0]["message"]["content"],
            raw_response=response,
            usage=dict(
                input_tokens=response["usage"]["prompt_tokens"],
                output_tokens=response["usage"]["completion_tokens"],
            ),
        )
