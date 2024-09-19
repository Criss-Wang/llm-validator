from abc import ABC, abstractmethod
from typing import List

from llm_validation.app.configs import ClientConfig


class Client(ABC):
    def __init__(self, config: ClientConfig):
        self.name = config.client_name
        self.base_url = config.model_base_url
        self.model_name = config.model_name
        self.model_options = config.model_options

    @abstractmethod
    async def predict_stream(self, messages: List):
        pass

    @abstractmethod
    async def predict(self, messages: List):
        pass
