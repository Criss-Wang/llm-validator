from setuptools import find_packages, setup

setup(
    name="llm-benchmark",
    version="1.0.0",
    description="LLM Model Benchmark",
    include_package_data=True,
    packages=find_packages(exclude=["tests"]),
    entry_points={
        "console_scripts": [
            "llm-benchmark = llm_benchmark.__main__:cli",
        ]
    },
)
