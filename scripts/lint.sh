#!/usr/bin/env bash
set -e
set -x

SRC=${1:-"src/"}
NOTEBOOKS=${@:-"*.ipynb"}

export ZENML_DEBUG=1
export ZENML_ANALYTICS_OPT_IN=false
flake8 $SRC
flake8-nb $NOTEBOOKS
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place $SRC --exclude=__init__.py --check
isort $SRC scripts --check-only
black $SRC $NOTEBOOKS --check
interrogate $SRC -c pyproject.toml
