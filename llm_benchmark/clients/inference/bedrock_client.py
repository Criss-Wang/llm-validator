import json
import logging
import time

import boto3

from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.data.data_model import DatasetRecord

logger = logging.getLogger(__name__)


class BedrockInferenceClient(LLMInferenceClient):
    def setup_client(self):
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")

    async def predict_stream(self, dataset: DatasetRecord):
        messages = [
            dict(
                role=message.role,
                content=message.content,
            )
            for message in dataset.messages
        ]

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {messages[0]['content']}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {messages[1]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        # Format the request payload using the model's native structure.
        request = {
            "prompt": prompt,
            # Optional inference parameters:
            "max_gen_len": 1024,
            "temperature": 0,
            "top_p": 1,
        }
        try:
            response_stream = self.client.invoke_model_with_response_stream(
                body=json.dumps(request),
                modelId=self._model_id,
            )
        except Exception as e:
            print(e)
            time.sleep(5)
            response_stream = self.client.invoke_model_with_response_stream(
                body=json.dumps(request),
                modelId=self._model_id,
            )
        for event in response_stream["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if "generation" in chunk:
                yield dict(
                    text=chunk["generation"],
                    raw_response=chunk,
                )
            else:
                continue
