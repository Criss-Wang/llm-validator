import asyncio
import re
from collections import Counter
from tqdm.asyncio import tqdm
from typing import Dict

from .base import AccuracyWithGroundTruth
from llm_validation.app.configs import MetricConfig, ClientConfig, PromptConfig
from llm_validation.components.factories.client_factory import init_client
from llm_validation.components.prompts import Prompt


class QuestionAnsweringAccuracy(AccuracyWithGroundTruth):
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
            name=var_map.get("prompt_name", "qa-judge"),
            path=var_map.get("prompt_path", "prompts/qa/prompt.yaml"),
            version=var_map.get("prompt_verions", 1),
        )
        self.client = init_client(client_config)
        self.prompt = Prompt(prompt_config)

    async def run_grading(self, results, include_labels):
        tasks = [
            self.grade((context, question), output, label)
            for context, question, output, label in zip(
                results.raw_inputs["context"],
                results.raw_inputs["question"],
                results.extracted_responses,
                results.labels,
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
            context=input[0], question=input[1], answer=output
        )
        try:
            result_content = await self.client.predict(messages)
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
        return "QuestionAnsweringAccuracy"
