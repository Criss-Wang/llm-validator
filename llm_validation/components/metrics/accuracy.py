from collections import defaultdict, Counter

import json
from sklearn.metrics import classification_report

from .base import Metric
from llm_validation.components.results import Result
from llm_validation.app.configs import MetricConfig, ClientConfig, PromptConfig
from llm_validation.components.clients import OpenAiClient
from llm_validation.components.prompts import Prompt


class AccuracyMetric(Metric):
    def measure(self, results: Result) -> None:
        self.scores = defaultdict(list)
        self.size = len(results)
        self.responses = results.responses
        self.labels = results.labels
        for input, output, label in zip(
            results.messages, results.responses, results.labels
        ):
            current_scores = self.grade(input, output, label)
            for score_name, score in current_scores.items():
                self.scores[score_name].append(score)

    def aggregate(self):
        raise NotImplementedError

    def grade(self, input, output, label):
        raise NotImplementedError

    def get_name(self):
        return "Accuracy"

    def get_stats(self):
        return self.stats

    def get_scores(self):
        return self.scores


class ClassificationAccuracy(AccuracyMetric):
    def grade(self, input, output: str, label: str):
        return {"correctness": output.lower() == label.lower()}

    def get_name(self):
        return "ClassificationAccuracy"

    def aggregate(self):
        correctness = self.scores["correctness"]
        self.stats.update(
            {
                "total_correct": sum(correctness),
                "total_wrong": len(correctness) - sum(correctness),
            }
        )
        self.stats.update(
            classification_report(self.labels, self.responses, output_dict=True)
        )


class CodeGenAccuracy(AccuracyMetric):
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        client_config = ClientConfig(
            name="openai",
            type="research",
            model_name="gpt-4o-mini",
            base_url="",
            model_options={"temperature": 0, "top_p": 1, "max_tokens": 1024},
        )
        prompt_config = PromptConfig(
            name="code-generation-judge",
            path="prompts/judge.yaml",
            version=1,
        )
        self.client = OpenAiClient(client_config)
        self.prompt = Prompt(prompt_config)

    def grade(self, input, output: str, label: str):
        messages = self.prompt.transform(
            generated_code_answer=output, expected_code_answer=label
        )
        try:
            result_content = self.client.sync_predict(messages)
            result_content = json.loads(result_content["text"])
            reason = result_content["reason"]
            code_quality = result_content["code_quality"]
            response_quality = result_content["response_quality"]
        except Exception as e:
            print(e)
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


# class FunctionCallMetric(AccuracyMetric):
#     need_text_response_match: List[bool] = []
#     text_response_similarity: List[float] = []
#     is_function_correct: List[bool] = []
#     is_function_params_correct: List[bool] = []
#     _model: SentenceTransformer

#     def __init__(self, model: str = "paraphrase-MiniLM-L6-v2", **kwargs):
#         super().__init__(**kwargs)

#         self._model = SentenceTransformer(model)

#     def grade(self, input, output: str, label: str):
#         try:
#             gen_response = ast.literal_eval(output)
#         except:
#             gen_response = {}
#         try:
#             exp_response = ast.literal_eval(label)
#         except:
#             exp_response = {}

#         need_cur_text_response_match = False
#         if (
#             gen_response.get("answer", "") == ""
#             and exp_response.get("answer", "") == ""
#         ) or (
#             gen_response.get("answer", "") != ""
#             and exp_response.get("answer", "") != ""
#         ):
#             need_cur_text_response_match = True
#         self.need_text_response_match.append(need_cur_text_response_match)

#     def process_record(self, record: LLMResultRecord) -> Dict[str, Any]:

#         import pdb

#         pdb.set_trace()
#         try:
#             gen_response = ast.literal_eval(record.response)
#         except:
#             gen_response = {}
#         try:
#             exp_response = ast.literal_eval(record.expected_response)
#         except:
#             exp_response = {}

#         need_cur_text_response_match = False
#         if (
#             gen_response.get("answer", "") == ""
#             and exp_response.get("answer", "") == ""
#         ) or (
#             gen_response.get("answer", "") != ""
#             and exp_response.get("answer", "") != ""
#         ):
#             need_cur_text_response_match = True
#         self.need_text_response_match.append(need_cur_text_response_match)

#         gen_func = gen_response.get("function", {"name": ""})
#         try:
#             exp_func = ast.literal_eval(exp_response.get("function", '{"name": ""}'))
#         except:
#             exp_func = {"name": ""}

#         is_cur_function_correct = gen_func.get("name", "") == exp_func.get("name", "")
#         self.is_function_correct.append(is_cur_function_correct)

#         is_cur_function_parameters_correct = gen_func.get(
#             "arguments", ""
#         ) == exp_func.get("arguments", "")
#         self.is_function_params_correct.append(is_cur_function_parameters_correct)

#         embedding1 = self._model.encode(
#             gen_response.get("answer", ""), convert_to_tensor=True
#         )
#         embedding2 = self._model.encode(
#             exp_response.get("answer", ""), convert_to_tensor=True
#         )

#         cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

#         self.text_response_similarity.append(cosine_similarity)

#         return {
#             self.get_id(): (
#                 need_cur_text_response_match,
#                 cosine_similarity,
#                 is_cur_function_correct,
#                 is_cur_function_parameters_correct,
#             )
#         }

#     def get_aggregated_metrics(self) -> Dict[str, Any]:
#         return {
#             f"{self.get_id()}_need_text_response_match": sum(
#                 self.need_text_response_match
#             )
#             / len(self.need_text_response_match),
#             f"{self.get_id()}_similarity_min": sum(self.text_response_similarity)
#             / len(self.text_response_similarity),
#             f"{self.get_id()}_is_function_correct": sum(self.is_function_correct)
#             / len(self.is_function_correct),
#             f"{self.get_id()}_is_function_params_correct": sum(
#                 self.is_function_params_correct
#             )
#             / len(self.is_function_params_correct),
#         }
