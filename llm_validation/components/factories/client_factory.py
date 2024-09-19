from llm_validation.app.configs import ClientType, ClientConfig
from llm_validation.components.clients import (
    Client,
    AnthropicClient,
    BedrockClient,
    OpenAiClient,
    TogetherClient,
    VertexAiClient,
    LocalClient,
)


def init_client(config: ClientConfig) -> Client:
    if config.client_type == ClientType.Local:
        return LocalClient(config)
    elif config.client_type == ClientType.ThirdPartyLLM:
        return init_api_client(config)
    else:
        raise ValueError(f"Client type not supported: {config.client_type}")


def init_api_client(config: ClientConfig) -> Client:
    if config.client_name == "anthropic":
        return AnthropicClient(config)
    elif config.client_name == "bedrock":
        return BedrockClient(config)
    elif config.client_name == "openai":
        return OpenAiClient(config)
    elif config.client_name == "together":
        return TogetherClient(config)
    elif config.client_name == "vertex":
        return VertexAiClient(config)
    else:
        raise ValueError(
            f"Third party API client type not supported: {config.client_name}"
        )
