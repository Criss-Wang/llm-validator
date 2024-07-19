from llm_benchmark.app.benchmark_orchestration import run_benchmark
from llm_benchmark.utilities.config_utils import load_benchmark_config

config = load_benchmark_config(filename="tests/configs/openai.json")
run_benchmark(config)
