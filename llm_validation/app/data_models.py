from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ClientType(str, Enum):
    VLLM = "vllm"
    OpenAI = "openai"
    LLMGateway = "llm-gateway"
    ResearchLLM = "research"


class AspectConfig(BaseModel):
    id: str
    params: Optional[Dict[str, Any]] = {}


class ClientConfig(BaseModel):
    type: ClientType
    config: Dict[str, Any] = {}


class EvaluatorConfig(BaseModel):
    id: str
    params: Optional[Dict[str, Any]] = {}


class PromptConfig(BaseModel):
    name: str
    tenant: str
    path: str
    version: int


class DatasetConfig(BaseModel):
    data_path: str


class TaskConfig(BaseModel):
    name: str


class LLMResultRecord(BaseModel):
    prompt: str
    request: Dict[str, Any]
    response: str
    expected_response: Optional[str] = None
    time_to_first_token: float
    total_time: float
    tokens_per_second: float
    number_of_tokens: int
    success: bool
    start_time: float
    end_time: float
    aspects: Dict[str, Any] = {}

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        self.aspects.update(metrics)


class ChatMessage(BaseModel):
    role: str
    content: str


class DatasetRecord(BaseModel):
    inputs: Dict[str, Any]
    messages: List[ChatMessage]
    expected_response: Optional[str] = None


class Prompt(BaseModel):
    pass
