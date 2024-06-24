from typing import Any, Dict

import requests

from llm_benchmark.aspects.aspect import Aspect
from llm_benchmark.data.data_model import LLMResultRecord


class LLMJudge(Aspect):
    counts: Dict[Any, int] = {}
    _host: str = "llm-gateway"
    _port: int = 7011
    _tenant_id: str
    _prompt_name: str
    _actual_response_field: str
    _expected_response_field: str
    _model_options: Dict

    def __init__(
        self,
        tenant: str,
        prompt_name: str,
        actual_response_field: str = None,
        expected_response_field: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._tenant_id = tenant
        self._prompt_name = prompt_name
        self._actual_response_field = actual_response_field
        self._expected_response_field = expected_response_field

        self._host = kwargs.get("host", self._host)
        self._port = kwargs.get("port", self._port)
        self._model_options = kwargs.get("model_options", {})

    def process_record(self, record: LLMResultRecord) -> Dict[str, Any]:
        params = []

        if self._actual_response_field:
            params.append(
                {"name": self._actual_response_field, "value": record.response}
            )

        if self._expected_response_field:
            params.append(
                {
                    "name": self._expected_response_field,
                    "value": record.expected_response,
                }
            )

        payload = {
            "tenantId": self._tenant_id,
            "promptName": self._prompt_name,
            "requestParams": params,
            "config": {
                "modelOptions": self._model_options,
            },
            "useCache": False,
            "debug": False,
        }

        try:
            uri = "http://{}:{}/llm-gateway/llm/v1/execution/fetchChatAnswer".format(
                self._host, self._port
            )
            response = requests.post(uri, json=payload, timeout=60)
            response.raise_for_status()

            result = str(response.json().get("message")).lower()

            if result not in ["true", "false"]:
                result = None
            else:
                result = result == "true"
        except Exception:
            result = None

        self.counts[result] = self.counts.get(result, 0) + 1

        return {
            self.get_id(): result,
        }

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        return {f"{self.get_id()}_{k}": v for k, v in self.counts.items()}
