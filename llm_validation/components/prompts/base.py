import yaml
from typing import List


from llm_validation.app.configs import PromptConfig
from llm_validation.utilities.prompt_utils import transform_prompt


class Prompt:
    def __init__(self, config: PromptConfig):
        self.name = config.name
        self.tenant = config.tenant
        self.messages = self.load_prompt(config)

    def load_prompt(self, config: PromptConfig):
        messages = []
        with open(config.path, "r+") as f:
            prompts = yaml.safe_load(f)
            for prompt in prompts:
                if prompt["name"] != config.name:
                    continue
                if "system" in prompt and prompt["system"]:
                    messages.append(
                        {
                            "role": "system",
                            "content": transform_prompt(prompt["system"]["value"]),
                        }
                    )
                if "user" in prompt and prompt["user"]:
                    messages.append(
                        {
                            "role": "user",
                            "content": transform_prompt(prompt["user"]["value"]),
                        }
                    )
        if not messages:
            raise ValueError("Unable to load prompt")
        return messages

    def transform(self, **inputs) -> List:
        final_messages = []
        for message in self.messages:
            complete_message = message.copy()
            complete_message["content"] = complete_message["content"].format(**inputs)
            final_messages.append(complete_message)
        return final_messages
