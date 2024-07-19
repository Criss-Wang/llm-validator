#!/bin/bash

tasks="chunk_validation"
models="anthropic anyscale-llama3-70b bedrock-llama3-70b gemini gpt4 monolyth-phi together-gemma together-llama3-70b together-mistral"

for task in ${tasks}
    do
    for model in ${models}
        do
            llm-validator run --config-path=configs/${task}/config-${model}.json --custom=True
        done
    done
