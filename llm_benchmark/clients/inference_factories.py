from llm_benchmark.clients.inference.inference_client import LLMInferenceClient
from llm_benchmark.clients.inference.anthropic_client import AnthropicInferenceClient
from llm_benchmark.clients.inference.anyscale_client import AnyscaleInferenceClient
from llm_benchmark.clients.inference.bedrock_client import BedrockInferenceClient
from llm_benchmark.clients.inference.gemini_client import GeminiInferenceClient
from llm_benchmark.clients.inference.monolyth_client import MonolythInferenceClient
from llm_benchmark.clients.inference.openai_client import OpenAIInferenceClient
from llm_benchmark.clients.inference.self_hosted_client import SelfHostedInferenceClient
from llm_benchmark.clients.inference.together_client import TogetherInferenceClient
from llm_benchmark.data.data_model import ClientConfig, ClientType


def create_inference_client(config: ClientConfig) -> LLMInferenceClient:
    client_class = {
        ClientType.AnthropicLLM: AnthropicInferenceClient,
        ClientType.AnyscaleLLM: AnyscaleInferenceClient,
        ClientType.BedrockLLM: BedrockInferenceClient,
        ClientType.GeminiLLM: GeminiInferenceClient,
        ClientType.MonolythLLM: MonolythInferenceClient,
        ClientType.OpenAILLM: OpenAIInferenceClient,
        ClientType.SelfHostedLLM: SelfHostedInferenceClient,
        ClientType.TogetherLLM: TogetherInferenceClient,
    }.get(config.type)

    if client_class is None:
        raise ValueError(f"Unsupported inference client type: {config.type}")

    return client_class(**config.config)
