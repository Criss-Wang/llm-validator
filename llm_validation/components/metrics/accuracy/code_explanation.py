import asyncio
import re
from collections import Counter
from tqdm.asyncio import tqdm

from .base import AccuracyWithGroundTruth
from llm_validation.app.configs import MetricConfig, ClientConfig, PromptConfig
from llm_validation.components.clients import OpenAiClient
from llm_validation.components.prompts import Prompt


class CodeExplanationAccuracy(AccuracyWithGroundTruth):
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
            name="code-explain-judge",
            path="prompts/code_explanation/prompt.yaml",
            version=1,
        )
        self.client = OpenAiClient(client_config)
        self.prompt = Prompt(prompt_config)

    async def run_grading(self, results, include_labels):
        tasks = [
            self.grade(input, output)
            for input, output in zip(
                results.raw_inputs["whole_func_string"], results.extracted_responses
            )
        ]

        graded_results = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Evaluation: {self.get_name()}",
        ):
            graded_results.append(await future)

        return graded_results

    async def grade(self, input, output: str, label: str = None):
        messages = self.prompt.transform(
            code_snippet=input,
            explanation=output,
        )
        try:
            result_content = self.client.sync_predict(messages)
            reason, score = self._parse_result(result_content["text"])
        except:
            print("error grading")
            reason = "error"
            score = -1
        return {
            "reason": reason,
            "score": score,
        }

    def _parse_result(self, content):
        score_pattern = re.compile(r"<score>(.*?)</score>", re.DOTALL)
        reason_pattern = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)

        score_match = score_pattern.search(content.lower())
        reason_match = reason_pattern.search(content.lower())

        score = score_match.group(1).strip() if score_match else -1
        reason = reason_match.group(1).strip() if reason_match else -1
        return reason, score

    def aggregate(self):
        code_score = self.scores["score"]
        self.stats.update(dict(Counter(code_score)))

    def get_name(self):
        return "CodeExplanationAccuracy"
