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
    if config.type == ClientType.Local:
        return LocalClient(config)
    elif config.type == ClientType.ResearchLLM:
        return init_research_client(config)
    else:
        raise ValueError(f"Client type not supported: {config.type}")


def init_research_client(config: ClientConfig) -> Client:
    if config.name == "anthropic":
        return AnthropicClient(config)
    elif config.name == "bedrock":
        return BedrockClient(config)
    elif config.name == "openai":
        return OpenAiClient(config)
    elif config.name == "together":
        return TogetherClient(config)
    elif config.name == "vertex":
        return VertexAiClient(config)
    else:
        raise ValueError(f"Research client type not supported: {config.name}")
