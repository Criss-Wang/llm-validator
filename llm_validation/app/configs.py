from enum import Enum
from typing import Any, List, Dict, Optional, Literal

from pydantic import BaseModel, ConfigDict


class ClientType(str, Enum):
    Local = "local"
    ThirdPartyLLM = "third_party_llm"


class TaskConfig(BaseModel):
    name: str


class ClientConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    client_type: ClientType
    client_name: str
    model_type: str
    model_name: str
    model_base_url: str = ""
    model_options: Dict = {}


class MetricConfig(BaseModel):
    type: Literal["cost", "latency", "accuracy", "security", "stability"]
    name: str = ""
    aspect: Optional[str] = ""
    kwargs: Optional[Dict] = {}


class PromptConfig(BaseModel):
    name: str
    tenant: Optional[str] = ""
    path: str
    version: int


class EvaluatorConfig(BaseModel):
    metrics: List[MetricConfig]


class DatasetConfig(BaseModel):
    data_path: str
    label_col: Optional[str] = None
    sanity_test: bool = False


class ExtractionConfig(BaseModel):
    type: str
    args: Dict[str, Any]


class ControllerConfig(BaseModel):
    parallelism: int = 12
    use_streaming: bool = False
    save_path: str
    extraction_config: Optional[ExtractionConfig] = None
    save_inference: bool = False


class ValidationConfig(BaseModel):
    project: str
    inference_only: bool = False

    task_config: TaskConfig
    client_config: ClientConfig
    prompt_config: PromptConfig
    evaluator_config: EvaluatorConfig
    dataset_config: DatasetConfig
    controller_config: ControllerConfig
