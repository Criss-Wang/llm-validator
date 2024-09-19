import json
from collections import Counter

from .base import AccuracyWithGroundTruth
from llm_validation.app.configs import MetricConfig, ClientConfig, PromptConfig
from llm_validation.components.clients import OpenAiClient
from llm_validation.components.prompts import Prompt


class CodeGenerationAccuracy(AccuracyWithGroundTruth):
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self.init_llm_as_judge()

    def init_llm_as_judge(self):
        client_config = ClientConfig(
            client_name="openai",
            client_type="third_party_llm",
            model_name="GPT4",
            model_type="gpt-4",
            model_options={
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 4096,
            },
        )
        prompt_config = PromptConfig(
            name="codegen-judge",
            path="prompts/judge/prompts.yml",
            version=1,
        )
        self.client = OpenAiClient(client_config)
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
            result_content = self.client.sync_predict(messages)
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
