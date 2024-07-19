import json
import time
import logging
from typing import List, Dict

import boto3

from llm_validation.components.clients import Client


class BedrockClient(Client):
    async def predict_stream(self, messages: List):
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {messages[0]['content']}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {messages[1]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        # Format the request payload using the model's native structure.
        request_payload = {"prompt": prompt}
        # Optional inference parameters:
        request_payload.update(**self.model_options)

        try:
            response_stream = client.invoke_model_with_response_stream(
                body=json.dumps(request_payload),
                modelId=self.model_name,
            )
        except Exception as e:
            print(e)
            time.sleep(5)
            response_stream = client.invoke_model_with_response_stream(
                body=json.dumps(request_payload),
                modelId=self.model_name,
            )
        for event in response_stream["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if "prompt_token_count" in chunk and chunk["prompt_token_count"]:
                self.input_tokens = chunk["prompt_token_count"]
            if "generation" in chunk:
                yield dict(
                    text=chunk["generation"],
                    raw_response=chunk,
                )
            else:
                continue

    async def predict(self, messages: List) -> Dict:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {messages[0]['content']}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {messages[1]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        # Format the request payload using the model's native structure.
        request_payload = {"prompt": prompt}
        # Optional inference parameters:
        request_payload.update(**self.model_options)

        try:
            response = client.invoke_model(
                body=json.dumps(request_payload),
                modelId=self.model_name,
            )
            response = json.loads(response["body"].read())
        except Exception as e:
            print(e)
            time.sleep(5)
            response = client.invoke_model(
                body=json.dumps(request_payload),
                modelId=self.model_name,
            )
            response = json.loads(response["body"].read())

        return dict(
            text=response["generation"],
            raw_response=response,
            usage=dict(
                input_tokens=response["prompt_token_count"],
                output_tokens=response["generation_token_count"],
            ),
        )

    def extract_usage(self, type: str = "input") -> int:
        if type == "input" and self.input_tokens:
            return self.input_tokens
