# LLM Validator

## Setup Guide

### Prerequisites

In order to run this project you will need python version 3.9/3.10/3.11 installed on your machine.

The project uses packages from the JFrog repository. To use the packages, you need to export the following environment variable:

### Setup environment

Clone repository:

```bash
git clone git@github.com:Criss-Wang/llm-validator.git
cd llm-validator
```

[*Optional*] Create and activate a virtual environment:

```bash
virtualenv -p python3.9 venv
. ./venv/bin/activate
```

Install the pre-commit hook:

```bash
pre-commit install
```

Install the project dependencies:

```bash
pip install -r requirements.dev.txt
pip install -e .
```

Login to your [Weights & Biases](https://wandb.ai/site) account:

```bash
wandb login
````

Initialize and export all the environment variables.
We support various third-party model inference providers. Please refer to `.env.sample` for a list of api keys required. then create an `.env` file and fill in the relevant api key values associate to each environment variable, and run 

```bash
export $(grep -v '^#' .env | xargs -0)
```

to export the environment variables.


### Basic usage
1. Define your prompt under `prompts` folder, in the following format
```yaml
- name: prompt-name
  system:
    value: >
      system prompt content here
  user:
    value: >
      user prompt content here
```
2. Import the validation dataset you'd like to use into `datasets` folder
3. Create a config file under `configs` folder. Refer to `configs/code_generation/openai.json` for how to structure your configurations in json format.


### Advanced usage
**Custom Client**

You can introduce additional client/api providers, or even local endpoints by implementing a `Client` defined under `llm_validation.component`.

**Custom Metrics**

You can introduce additional metrics by implementing a `Metric` or inheriting from one of the 5 domains (Cost, Latency, Accuracy, Security, Stability) under `llm_validation.component`.



## Tutorials
- [Model Validation Blog](https://criss-wang.com/post/software/model-iteration-research-validation/)

