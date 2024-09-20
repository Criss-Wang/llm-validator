import ast
import json

import numpy as np


from .base import AccuracyWithGroundTruth


class FunctionCallingAccuracy(AccuracyWithGroundTruth):
    async def grade(self, input, output: str, label: str):
        function_accuracy, argument_correctness = -1, -1
        try:
            try:
                label_json = json.loads(label)
                output_json = json.loads(output)
            except:
                label_json = ast.literal_eval(label)
                output_json = ast.literal_eval(output)
            function_accuracy = label_json["name"] == output_json["name"]
            argument_correctness = label_json["parameters"] == output_json["parameters"]
        except Exception as e:
            print(e)

        return {
            "function_accuracy": function_accuracy,
            "argument_correctness": argument_correctness,
        }

    def get_name(self):
        return "FunctionCallingAccuracy"

    def aggregate(self):
        function_accuracy = [
            score for score in self.scores["function_accuracy"] if score != -1
        ]
        argument_correctness = [
            score for score in self.scores["argument_correctness"] if score != -1
        ]
        self.stats.update(
            {
                "function_accuracy": sum(function_accuracy) / len(function_accuracy),
                "argument_correctness": sum(argument_correctness)
                / len(argument_correctness),
            }
        )
        # self.stats.update(
        #     {
        #         "total_correct": sum(correctness),
        #         "total_wrong": len(correctness) - sum(correctness),
        #     }
        # )
        # self.stats.update(
        #     classification_report(self.labels, self.responses, output_dict=True)
        # )
