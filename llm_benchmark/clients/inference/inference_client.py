from abc import ABC

from llm_benchmark.data.data_model import DatasetRecord, PromptConfig


class LLMInferenceClient(ABC):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        type: str = "anyscale",
        model_options: dict = None,
        **kwargs,
    ):
        self._model_id = model_id
        self._base_url = base_url
        self._model_options = model_options or {}
        self._type = type
        self.kwargs = kwargs
        self.client = self.setup_client()

    async def predict_stream(self, prompt: PromptConfig, dataset: DatasetRecord):
        raise NotImplementedError

    def setup_client(self):
        raise NotImplementedError

    def predict(self, dataset: DatasetRecord):
        raise NotImplementedError
