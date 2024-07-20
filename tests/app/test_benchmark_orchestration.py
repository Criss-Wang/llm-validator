from llm_validation.app.orchestration import run_validation
from llm_validation.utilities.config_utils import load_validation_config

config = load_validation_config(filename="tests/configs/openai.json")
run_validation(config)
