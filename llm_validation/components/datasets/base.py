import pandas as pd

from llm_validation.app.configs import DatasetConfig
from llm_validation.components.prompts import Prompt


class Dataset:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = self.load_data(config)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data["messages"]):
            messages = self.data["messages"][self.index]
            # expected_response = self.data["expected_responses"][self.index]
            self.index += 1
            return messages
        else:
            # No more items to iterate over
            raise StopIteration

    def load_data(self, config: DatasetConfig):
        data = pd.read_csv(config.data_path, index_col=0)
        inputs = data.to_dict(orient="records")
        expected_responses = data[config.label_col].tolist()
        return {"inputs": inputs, "expected_responses": expected_responses}

    def adopt_prompt(self, prompt: Prompt):
        self.data["messages"] = [
            prompt.transform(**input_vars) for input_vars in self.data["inputs"]
        ]
        return self

    def get_labels(self):
        return self.data["expected_responses"]
