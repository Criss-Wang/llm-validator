from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ClientType(str, Enum):
    AnthropicLLM = "anthropic"
    AnyscaleLLM = "anyscale"
    BedrockLLM = "bedrock"
    GeminiLLM = "gemini"
    MonolythLLM = "monolyth"
    OpenAILLM = "openai"
    SelfHostedLLM = "self-hosted"
    TogetherLLM = "together"


class AspectConfig(BaseModel):
    id: str
    params: Optional[Dict[str, Any]] = {}


class ClientConfig(BaseModel):
    type: ClientType
    config: Dict[str, Any] = {}


class PromptConfig(BaseModel):
    name: str
    tenant: str
    path: str
    version: int


class BenchMarkConfig(BaseModel):
    project: str
    task: str
    dataset_path: str
    model: str
    prompt: PromptConfig
    aspects: List[AspectConfig]
    client: ClientConfig
    parallelism: int = 12
    label_col: Optional[str] = None


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
