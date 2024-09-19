import ast

import numpy as np
from sentence_transformers import SentenceTransformer, util


from .base import AccuracyWithGroundTruth
from llm_validation.app.configs import MetricConfig


class FunctionCallMetric(AccuracyWithGroundTruth):
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self._model = SentenceTransformer(config.kwargs["model"])

    async def grade(self, input, output: str, label: str):
        try:
            gen_response = ast.literal_eval(output)
        except:
            gen_response = {}
        try:
            exp_response = ast.literal_eval(label)
        except:
            exp_response = {}

        # evaluate answer section existence match
        answer_section_existence_match = False
        if (
            gen_response.get("answer", "") == ""
            and exp_response.get("answer", "") == ""
        ) or (
            gen_response.get("answer", "") != ""
            and exp_response.get("answer", "") != ""
        ):
            answer_section_existence_match = True

        # evaluate function correctness
        gen_func = gen_response.get("function", {"name": ""})
        try:
            exp_func = ast.literal_eval(exp_response.get("function", '{"name": ""}'))
        except:
            exp_func = {"name": ""}
        function_correctness = gen_func.get("name", "") == exp_func.get("name", "")

        # evaluate function parameters correctness
        function_parameters_correctness = gen_func.get("arguments", "") == exp_func.get(
            "arguments", ""
        )

        # evaluate answer similarity
        embedding1 = self._model.encode(
            gen_response.get("answer", ""), convert_to_tensor=True
        )
        embedding2 = self._model.encode(
            exp_response.get("answer", ""), convert_to_tensor=True
        )

        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

        return {
            "answer_section_existence_match": answer_section_existence_match,
            "function_correctness": function_correctness,
            "function_parameters_correctness": function_parameters_correctness,
            "similarity_score": cosine_similarity,
        }

    def aggregate(self):
        cosine_similarities = self.scores["similarity_score"]
        function_correctness = self.scores["function_correctness"]
        function_parameters_correctness = self.scores["function_parameters_correctness"]
        answer_section_existence_match = self.scores["answer_section_existence_match"]
        self.stats.update(
            {
                "answer_section_existence_match_pct": np.mean(
                    answer_section_existence_match
                ),
                "function_correct_pct": np.mean(function_correctness),
                "function_parameters_correct_pct": np.mean(
                    function_parameters_correctness
                ),
            }
        )
        self.stats.update(
            {
                "similarity_score_mean": np.mean(cosine_similarities),
                "similarity_score_max": np.max(cosine_similarities),
                "similarity_score_min": np.min(cosine_similarities),
                "similarity_score_p75": np.percentile(cosine_similarities, 75),
                "similarity_score_p25": np.percentile(cosine_similarities, 25),
            }
        )
