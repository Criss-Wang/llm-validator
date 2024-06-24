import logging
import re
import os
import yaml
import numpy as np
from copy import deepcopy
from ast import literal_eval
from typing import List, Dict, Any

import openai
from llm_benchmark.aspects.aspect import Aspect
from llm_benchmark.data.data_model import LLMResultRecord

logger = logging.getLogger(__name__)


class Generation(Aspect):
    scores: List[int] = []
    expected_scores: List[int] = []

    def __init__(self, prompt_name: str, method: str = "llm_score", **kwargs):
        super().__init__(**kwargs)
        self._prompt = self.load_prompt(prompt_name)
        self._method = method

    def process_record(self, record: LLMResultRecord) -> Dict[str, Any]:
        expected_response = literal_eval(record.expected_response)["intent"]
        try:
            pattern = r"\{[\s\S]*?\}"
            response = literal_eval(re.findall(pattern, record.response)[0])[
                "searchQuery"
            ]
        except Exception as e:
            print(e)
            print(record.response)
            response = record.response
        score = self.get_llm_judge_score(record.request["inputs"]["query"], response)

        expected_score = self.get_llm_judge_score(
            record.request["inputs"]["query"], expected_response
        )
        self.scores.append(score)
        self.expected_scores.append(expected_score)

        return {self.get_id(): score / expected_score if expected_score > 0 else 1.0}

    def transform_prompt(self, raw_prompt: str) -> str:
        # replace single curly with double curly
        prompt_str = raw_prompt.replace("{", "{{").replace("}", "}}")

        # replace $var pattern with {var} pattern
        idx_pattern = r'"\$([a-z_A-Z]+)"'
        idx_pattern = re.compile(idx_pattern)
        return idx_pattern.sub(r'"{\1}"', prompt_str)

    def load_prompt(self, prompt_name: str) -> Dict:
        with open("prompts/judge/prompts.yml") as f:
            prompts = yaml.load(f, Loader=yaml.SafeLoader)

        for curr in prompts:
            if curr["name"] == prompt_name:
                prompt = [
                    {
                        "role": "system",
                        "content": self.transform_prompt(curr["system"]["value"]),
                    },
                    {
                        "role": "user",
                        "content": self.transform_prompt(curr["user"]["value"]),
                    },
                ]
                return prompt
        return []

    def get_llm_judge_score(self, request: str, response: str) -> int:
        messages = deepcopy(self._prompt)
        inputs = {"QUERY": request, "SUMMARY": response}
        messages[1]["content"] = messages[1]["content"].format(**inputs)

        client = openai.AzureOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("OPENAI_API_BASE"),
        )
        try:
            response = client.chat.completions.create(
                model="GPT4", messages=messages, temperature=0, max_tokens=512, top_p=1
            )
            response = response.choices[0].message.content.strip()
            pattern = r"<score>(.*)<\/score>"
            eval_score = re.findall(pattern, response)[0].strip()
        except Exception as e:
            print(e)
            eval_score = 0
        return int(eval_score)

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        return {
            f"{self.get_id()}_actual_score_mean": np.mean(self.scores),
            f"{self.get_id()}_actual_score_max": max(self.scores),
            f"{self.get_id()}_actual_score_min": min(self.scores),
            f"{self.get_id()}_actual_score_p90": np.percentile(self.scores, 10),
            f"{self.get_id()}_expected_score_mean": np.mean(self.expected_scores),
            f"{self.get_id()}_expected_score_max": max(self.expected_scores),
            f"{self.get_id()}_expected_score_min": min(self.expected_scores),
            f"{self.get_id()}_expected_score_p90": np.percentile(
                self.expected_scores, 10
            ),
        }
