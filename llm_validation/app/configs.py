from enum import Enum
from typing import Any, List, Dict, Optional, Literal

from pydantic import BaseModel, ConfigDict


class ClientType(str, Enum):
    VLLM = "vllm"
    OpenAI = "openai"
    LLMGateway = "llm-gateway"
    ResearchLLM = "research"


class TaskConfig(BaseModel):
    name: str


class ClientConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    type: ClientType
    model_name: str
    base_url: str = ""
    model_options: Dict = {}


class JudgeConfig(BaseModel):
    llm_client: ClientConfig


class MetricConfig(BaseModel):
    type: Literal["cost", "latency", "accuracy", "security", "stability"]
    aspect: Optional[str] = ""
    kwargs: Optional[Dict] = {}


class EvaluatorConfig(BaseModel):
    metrics: List[MetricConfig]
    llm_judge: Optional[JudgeConfig] = None


class PromptConfig(BaseModel):
    name: str
    tenant: str
    path: str
    version: int


class DatasetConfig(BaseModel):
    data_path: str
    label_col: Optional[str] = None


class ControllerConfig(BaseModel):
    parallelism: int = 12
    use_streaming: bool = False
    save_path: str


class ValidationConfig(BaseModel):
    project: str

    task_config: TaskConfig
    client_config: ClientConfig
    prompt_config: PromptConfig
    evaluator_config: EvaluatorConfig
    dataset_config: DatasetConfig
    controller_config: ControllerConfig
