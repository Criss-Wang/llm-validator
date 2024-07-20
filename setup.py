from setuptools import find_packages, setup

setup(
    name="llm-validator",
    version="1.0.0",
    description="LLM Model Validation",
    include_package_data=True,
    packages=find_packages(exclude=["tests"]),
    entry_points={
        "console_scripts": [
            "llm-validator = llm_validation.app.orchestration:cli",
        ]
    },
)
