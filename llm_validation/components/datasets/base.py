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
        """
        Note: make sure the column name aligns with the prompt variable naming
        """
        data = pd.read_csv(config.data_path, index_col=0)

        # Only use the first 3 rows for sanity testing
        if config.sanity_test:
            data = data.head(3)

        inputs = data.to_dict(orient="records")

        if config.label_col:
            if config.label_col not in data:
                raise ValueError(
                    f"Label column {config.label_col} not found in dataset."
                )
            expected_responses = data[config.label_col].tolist()
        else:
            expected_responses = [None] * len(inputs)

        return {"inputs": inputs, "expected_responses": expected_responses}

    def adopt_prompt(self, prompt: Prompt):
        try:
            self.data["messages"] = [
                prompt.transform(**input_vars) for input_vars in self.data["inputs"]
            ]
        except:
            import pdb

            pdb.set_trace()
        return self

    def get_inputs_df(self):
        return pd.DataFrame(self.data["inputs"])

    def get_raw_inputs(self):
        df = pd.DataFrame(self.data["inputs"])
        return df.to_dict(orient="list")

    def get_labels(self):
        return self.data["expected_responses"]
