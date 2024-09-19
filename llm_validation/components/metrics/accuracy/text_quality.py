import re
import json
from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import AccuracyWithoutGroundTruth, AccuracyWithGroundTruth
from llm_validation.app.configs import MetricConfig


class RegexMatch(AccuracyWithoutGroundTruth):
    def __init__(self, config: MetricConfig):
        super().__init__(config)

        regex_list = config["regex_list"]
        self._regex = [re.compile(r) for r in regex_list]

    async def grade(self, input, output: str, label: str):
        try:
            regex_matchs = [bool(re.search(regex, output)) for regex in self._regex]
            return {
                "regex_match_any": any(regex_matchs),
                "regex_match_all": all(regex_matchs),
            }
        except Exception as e:
            print(f"output: {output} regex list: {self._regex}")
            print(e)

    def get_name(self):
        return "RegexMatch"

    def aggregate(self):
        self.stats.update(
            {
                "regex_match_any_pct": np.mean(self.scores["regex_match_any"]),
                "regex_match_all_pct": np.mean(self.scores["regex_match_all"]),
            }
        )


class JsonCorrectnessMetric(AccuracyWithGroundTruth):
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self._expected_fields = SentenceTransformer(config.kwargs["expected_fields"])

    async def grade(self, input, output: str, label: str):
        dict_actual = self._parse_json(output)
        dict_expected = self._parse_json(label)
        valid_json = dict_actual and dict_expected

        match_all, match_any, mismatch_fields, missing_fields = (
            self._compare_json_values(dict_actual, dict_expected)
        )
        return {
            "match_all": match_all,
            "match_any": match_any,
            "mismatch_fields": mismatch_fields,
            "missing_fields": missing_fields,
            "valid_json": valid_json,
        }

    def _parse_json(self, json_str: str):
        try:
            return json.loads(json_str)
        except Exception:
            return None

    def _compare_json_values(self, dict_actual: Dict, dict_expected: Dict):
        if not dict_actual or not dict_expected:
            return False, False, "Invalid Json", "Invalid Json"

        match_any = False
        match_all = True
        mismatch_fields = []
        missing_fields = []

        for field in self._expected_fields:
            actual_value = dict_actual.get(field, "missing")
            expected_value = dict_expected.get(field, "missing")
            if actual_value == "missing":
                missing_fields.append(field)
            if actual_value != expected_value:
                match_all = False
                mismatch_fields.append(
                    f"actual: {actual_value} expected: {expected_value}"
                )
            else:
                match_any = True
        mismatch_fields = "\n".join(mismatch_fields)
        missing_fields = ", ".join(missing_fields)
        return match_all, match_any, mismatch_fields, missing_fields

    def get_name(self):
        return "JsonCorrectnessMetric"

    def aggregate(self):
        match_all = np.array(self.scores["match_all"])
        match_any = np.array(self.scores["match_any"])
        valid_json = np.array(self.scores["valid_json"])

        match_all_valid = match_all[valid_json]
        match_any_valid = match_any[valid_json]

        self.stats.update(
            {
                "match_all_valid_only_pct": np.mean(match_all_valid),
                "match_all_full_pct": np.mean(match_all),
                "match_any_valid_only_pct": np.mean(match_any_valid),
                "match_any_full_pct": np.mean(match_any),
                "valid_json_pct": np.mean(valid_json),
            }
        )
