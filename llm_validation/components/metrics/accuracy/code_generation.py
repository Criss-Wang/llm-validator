import json
from collections import Counter
from typing import Dict

from .base import AccuracyWithGroundTruth
from llm_validation.app.configs import MetricConfig, ClientConfig, PromptConfig
from llm_validation.components.factories.client_factory import init_client
from llm_validation.components.prompts import Prompt


class CodeGenerationAccuracy(AccuracyWithGroundTruth):
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self.init_llm_as_judge(config.kwargs)

    def init_llm_as_judge(self, var_map: Dict):
        client_config = ClientConfig(
            client_name=var_map.get("client_name", "anthropic"),
            client_type=var_map.get("client_type", "third_party_llm"),
            model_name=var_map.get("model_name", "claude-3-5-sonnet-20240620"),
            model_type=var_map.get("model_type", "claude-3.5-sonnet"),
            model_options=var_map.get("model_options", {}),
        )
        prompt_config = PromptConfig(
            name=var_map.get("prompt_name", "code-generation-judge"),
            path=var_map.get("prompt_path", "prompts/code_generation/prompt.yaml"),
            version=var_map.get("prompt_verions", 1),
        )
        self.client = init_client(client_config)
        self.prompt = Prompt(prompt_config)

    def get_name(self):
        return "CodeGenerationAccuracy"

    async def grade(self, input, output: str, label: str):
        messages = self.prompt.transform(
            user_request=str(input),
            generated_code_answer=output,
            expected_code_answer=label,
        )
        try:
            result_content = await self.client.predict(messages)
            result_content = json.loads(result_content["text"])
            reason = result_content["reason"]
            code_quality = result_content["code_quality"]
            response_quality = result_content["response_quality"]
        except:
            print("error")
            reason = "error"
            code_quality = "wrong"
            response_quality = "bad"
        return {
            "reason": reason,
            "code_quality": code_quality,
            "response_quality": response_quality,
        }

    def aggregate(self):
        code_quality = self.scores["code_quality"]
        response_quality = self.scores["response_quality"]
        self.stats.update(dict(Counter(code_quality)))
        self.stats.update(dict(Counter(response_quality)))
