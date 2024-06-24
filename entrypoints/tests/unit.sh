#!/usr/bin/env bash

set -e

pip install --quiet -r requirements.dev.txt
python -m coverage run -m pytest tests/unit
python -m coverage report
