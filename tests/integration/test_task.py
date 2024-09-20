import os


# Run a command
for client in [
    # "chunk_validation",
    # "code_explanation",
    # "code_generation",
    # "function_calling",
    # "ner",
    # "qa",
    "summarization",
]:
    _ = os.system(
        f"llm-validator run --config-path=tests/integration/configs/task_test/{client}.json"
    )
