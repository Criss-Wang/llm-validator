from abc import ABC, abstractmethod
from typing import List, Dict, Any

import pandas as pd


class Result(ABC):
    @abstractmethod
    def to_df(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class InferenceResult(Result):
    def __init__(self, results: List[Dict], labels: List[Any]):
        self.results = results
        self.successes = [r["success"] for r in results]
        self.messages = [r["messages"] for r in results]
        self.responses = [r["response"] for r in results]
        self.token_statistics = [r["token_statistics"] for r in results]
        self.time_statistics = [r["time_statistics"] for r in results]
        self.labels = labels

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "messages": self.messages,
                "response": self.responses,
                "label": self.labels,
                "success": self.successes,
            },
        )

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
