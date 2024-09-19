import os


# Run a command
for client in [
    # "anthropic",
    "openai",
    # "together",
    # "vertexai",
    # "local_phi3",
]:
    _ = os.system(
        f"llm-validator run --config-path=tests/integration/configs/client_test/{client}.json"
    )
