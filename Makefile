ifneq (,$(wildcard ./.env))
    include .env
    export
endif

.SILENT:
.PHONY: refresh install setup lint test docs

refresh:

build: install setup test

setup:
	echo "=== Port forward the required services ==="
	export $(grep -v '^#' .env | xargs -0)
	docker pull ghcr.io/mlflow/mlflow
	docker run -d \
		-p 5000:5000 \
		-e MLFLOW_TRACKING_URI=http://0.0.0.0:5000 \
		-v $(pwd)/mlruns:/mlflow/mlruns \
		ghcr.io/mlflow/mlflow:latest

test:
	echo "=== Run tests ==="
	if ! command -v pytest &> /dev/null; then \
        echo "Installing pytest..."; \
        pip install pytest; \
    fi
	# export TEST_ENV=dev && \
	# python -m pytest

docs:

update_env:
	export $(grep -v '^#' .env | xargs -0)

	python -m flake8 llm_benchmark tests

kill_k:
	ps aux | grep -i kubectl | grep -v grep | awk {'print $$2'} | xargs kill
