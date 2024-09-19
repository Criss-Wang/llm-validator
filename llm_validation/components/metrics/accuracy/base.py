import asyncio
from tqdm.asyncio import tqdm

from ..base import Metric
from llm_validation.components.results import Result


class AccuracyMetric(Metric):
    def get_stats(self):
        return self.stats

    def get_scores(self):
        return self.scores

    async def grade(self, input, output, label=None):
        raise NotImplementedError

    async def run_grading(self, results, include_labels):
        tasks = (
            [
                (
                    self.grade(input, output, label)
                    if include_labels
                    else self.grade(input, output)
                )
                for input, output, label in zip(
                    results.messages, results.extracted_responses, results.labels
                )
            ]
            if include_labels
            else [
                self.grade(input, output)
                for input, output in zip(results.messages, results.extracted_responses)
            ]
        )

        graded_results = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Evaluation: {self.get_name()}",
        ):
            graded_results.append(await future)

        return graded_results

    def update_scores(self, grades):
        for current_scores in grades:
            for score_name, score in current_scores.items():
                self.scores[score_name].append(score)


class AccuracyWithGroundTruth(AccuracyMetric):
    def measure(self, results: Result):
        self.size = len(results)
        self.labels = results.labels
        self.responses = results.extracted_responses
        grades = asyncio.run(self.run_grading(results, include_labels=True))
        self.update_scores(grades)


class AccuracyWithoutGroundTruth(AccuracyMetric):
    def measure(self, results: Result):
        self.size = len(results)
        self.labels = results.labels
        self.responses = results.extracted_responses
        grades = asyncio.run(self.run_grading(results, include_labels=False))
        self.update_scores(grades)
