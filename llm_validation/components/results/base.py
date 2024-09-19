from abc import ABC, abstractmethod
from typing import List, Dict, Any

import pandas as pd

from llm_validation.components.datasets import Dataset
from llm_validation.components.evaluators.extractors import create_extractor
from llm_validation.app.configs import ExtractionConfig


class Result(ABC):
    @abstractmethod
    def to_df(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class InferenceResult(Result):
    def __init__(
        self,
        dataset: Dataset,
        results: List[Dict],
        labels: List[Any],
        extraction_config: ExtractionConfig = None,
    ):
        self.results = results
        self.raw_inputs = dataset.get_raw_inputs()
        self.inputs_df = dataset.get_inputs_df()
        self.successes = [r["success"] for r in results]
        self.messages = [r["messages"] for r in results]
        self.responses = [r["response"].strip() for r in results]
        self.extracted_responses = self._extract_answers(extraction_config)
        self.token_statistics = [r["token_statistics"] for r in results]
        self.time_statistics = [r["time_statistics"] for r in results]
        self.labels = labels

    def _extract_answers(self, extraction_config: ExtractionConfig) -> List[str]:
        answer_extractor = create_extractor(extraction_config)
        return [answer_extractor(response) for response in self.responses]

    def to_df(self) -> pd.DataFrame:
        inference_df = pd.DataFrame(
            {
                "messages": self.messages,
                "response": self.responses,
                "extracted_response": self.extracted_responses,
                "label": self.labels,
                "success": self.successes,
            },
        )
        return pd.concat([self.inputs_df, inference_df], axis=1)

    def __len__(self) -> int:
        return len(self.results)


class EvaluationResult(Result):
    def __init__(self, evaluation_scores: Dict, aggregated_metrics: Dict):
        self.evaluation_scores = evaluation_scores
        self.aggregated_metrics = aggregated_metrics

    def to_df(self) -> pd.DataFrame:
        final_stats = {}
        for metric, score_dict in self.evaluation_scores.items():
            for score_name, score in score_dict.items():
                final_stats[f"{metric}_{score_name}"] = score
        return pd.DataFrame(final_stats)

    def __len__(self) -> int:
        return len(self.evaluation_scores["outputs"])
