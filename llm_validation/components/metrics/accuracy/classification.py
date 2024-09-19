import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util


from .base import AccuracyWithGroundTruth
from llm_validation.app.configs import MetricConfig


class ClassificationAccuracy(AccuracyWithGroundTruth):
    async def grade(self, input, output: str, label: str):
        return {"correctness": output.lower().strip() == str(label).lower().strip()}

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


class CosineSimilarity(AccuracyWithGroundTruth):
    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self._model = SentenceTransformer(config.kwargs["model"])

    async def grade(self, input, output: str, label: str):
        embedding1 = self._model.encode(output, convert_to_tensor=True)
        embedding2 = self._model.encode(label, convert_to_tensor=True)

        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

        return {"similarity_score": cosine_similarity}

    def get_name(self):
        return "CosineSimilarity"

    def aggregate(self):
        cosine_similarities = self.scores["similarity_score"]
        self.stats.update(
            {
                "similarity_score_mean": np.mean(cosine_similarities),
                "similarity_score_max": np.max(cosine_similarities),
                "similarity_score_min": np.min(cosine_similarities),
                "similarity_score_p75": np.percentile(cosine_similarities, 75),
                "similarity_score_p25": np.percentile(cosine_similarities, 25),
            }
        )
        self.stats.update(
            classification_report(self.labels, self.responses, output_dict=True)
        )
