import asyncio
import re
from collections import Counter
from tqdm.asyncio import tqdm
from typing import Dict

from .base import AccuracyWithoutGroundTruth
from llm_validation.app.configs import MetricConfig, ClientConfig, PromptConfig
from llm_validation.components.factories.client_factory import init_client
from llm_validation.components.prompts import Prompt


class SummarizationAccuracy(AccuracyWithoutGroundTruth):
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
        self.client = init_client(client_config)

        prompt_config = PromptConfig(
            name=var_map.get("accuracy_prompt_name", "qa-judge"),
            path=var_map.get("prompt_path", "prompts/qa/prompt.yaml"),
            version=var_map.get("accuracy_prompt_verions", 1),
        )
        self.accuracy_prompt = Prompt(prompt_config)
        prompt_config = PromptConfig(
            name=var_map.get("adherence_prompt_name", "qa-judge"),
            path=var_map.get("prompt_path", "prompts/qa/prompt.yaml"),
            version=var_map.get("adherence_prompt_verions", 1),
        )
        self.adherence_prompt = Prompt(prompt_config)
        prompt_config = PromptConfig(
            name=var_map.get("quality_prompt_name", "qa-judge"),
            path=var_map.get("prompt_path", "prompts/qa/prompt.yaml"),
            version=var_map.get("quality_prompt_verions", 1),
        )
        self.quality_prompt = Prompt(prompt_config)

    async def run_grading(self, results, include_labels):
        tasks = [
            self.grade(input, output)
            for input, output in zip(
                results.raw_inputs["record"],
                results.extracted_responses,
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
        messages = self.accuracy_prompt.transform(
            original_content=input, summary=output
        )
        try:
            result_content = await self.client.predict(messages)
            accuracy_reason, accuracy_score = self._parse_result(result_content["text"])
        except:
            print("error grading")
            accuracy_reason = "error"
            accuracy_score = -1

        messages = self.adherence_prompt.transform(
            original_content=input, summary=output
        )
        try:
            result_content = await self.client.predict(messages)
            adherence_reason, adherence_score = self._parse_result(
                result_content["text"]
            )
        except:
            print("error grading")
            adherence_reason = "error"
            adherence_score = -1

        messages = self.quality_prompt.transform(original_content=input, summary=output)
        try:
            result_content = await self.client.predict(messages)
            quality_reason, quality_score = self._parse_result(result_content["text"])
        except:
            print("error grading")
            quality_reason = "error"
            quality_score = -1

        return {
            "accuracy_reason": accuracy_reason,
            "accuracy_score": accuracy_score,
            "adherence_reason": adherence_reason,
            "adherence_score": adherence_score,
            "quality_reason": quality_reason,
            "quality_score": quality_score,
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
        self.stats.update(
            {
                f"accuracy_score={score}": cnt
                for score, cnt in dict(Counter(self.scores["accuracy_score"])).items()
            }
        )
        self.stats.update(
            {
                f"adherence_score={score}": cnt
                for score, cnt in dict(Counter(self.scores["adherence_score"])).items()
            }
        )
        self.stats.update(
            {
                f"quality_score={score}": cnt
                for score, cnt in dict(Counter(self.scores["quality_score"])).items()
            }
        )

    def get_name(self):
        return "SummarizationAccuracy"
