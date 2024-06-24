from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util

from llm_benchmark.aspects.aspect import Aspect
from llm_benchmark.data.data_model import LLMResultRecord


class CosineSimilarity(Aspect):
    results: List[float] = []
    _model: SentenceTransformer

    def __init__(self, model: str = "paraphrase-MiniLM-L6-v2", **kwargs):
        super().__init__(**kwargs)

        self._model = SentenceTransformer(model)

    def process_record(self, record: LLMResultRecord) -> Dict[str, Any]:
        embedding1 = self._model.encode(record.response, convert_to_tensor=True)
        embedding2 = self._model.encode(
            record.expected_response, convert_to_tensor=True
        )

        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

        self.results.append(cosine_similarity)

        return {self.get_id(): cosine_similarity}

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        return {
            f"{self.get_id()}_mean": sum(self.results) / len(self.results),
            f"{self.get_id()}_max": max(self.results),
            f"{self.get_id()}_min": min(self.results),
        }
