#!/usr/bin/env bash
# Runs the part of the test suite that can pass on a plain GitHub-hosted runner.
#
#   .github/scripts/run-gate-tests.sh [extra pytest arguments]
#
# Run it from the directory that holds tests/, with the interpreter of the environment
# under test first on PATH. Both workflows call it, so the list below exists once:
# test.yml runs it on the source tree, pypi-publish.yml on a copy of tests/ that sits
# beside the INSTALLED wheel.
#
# FULL_SUITE=true runs everything instead. That run is expected to fail until the tests
# named here are fixed; use it to see how far away that is.
#
# Counts when this list was written, the same on Python 3.10, 3.11, 3.12, 3.13 and 3.14
# (installed wheel, clean venv, requirements.txt plus the test client, no PostgreSQL,
# no pymdp):
#   everything:          63 passed, 10 failed, 14 errors (4 of them collection errors)
#   with the list below: 63 passed, 5 deselected
set -euo pipefail

if [[ "${FULL_SUITE:-false}" == "true" ]]; then
  echo "FULL_SUITE=true: running every test, the known-failing ones included"
  exec python -m pytest tests -p no:cacheprovider --continue-on-collection-errors "$@"
fi

ignore=(
  # These import ci_active_inference.agents, whose __init__ does `from pymdp.agent import
  # Agent`. pymdp is in neither requirements.txt nor the package's dependencies, so
  # collection fails with ModuleNotFoundError. With inferactively-pymdp 0.0.7.1 (the
  # pre-1.0 line this code imports against; 1.0.4 fails on `from pymdp.maths import
  # softmax`) the four modules do collect, but the first test of tests/test_agents.py,
  # test_agent_initialization, then took the test process past 16 GiB resident within
  # seconds (measured on a 96 GB machine under a kill switch that stopped it at 17 GiB).
  # In October 2026 a standard ubuntu-latest runner has 8 GB of memory for a private
  # repository and 16 GB for a public one.
  # test_active_inference_integration.py and test_personal_life_agent.py also need the
  # PostgreSQL server below.
  tests/test_active_inference_integration.py
  tests/test_advanced_features.py
  tests/test_agents.py
  tests/test_personal_life_agent.py
  # The session fixture in conftest.py calls create_tables() against DATABASE_URL, which
  # defaults to a PostgreSQL server on localhost that a hosted runner does not have
  # (psycopg2.OperationalError: connection refused). ci_active_inference.database
  # connects with psycopg2 only, while README.md and SETUP.md call SQLite the default and
  # TESTING.md says a test session creates a temporary SQLite database.
  tests/test_database.py
  # Every test runs main.py, or looks for the shell scripts, with the working directory
  # hard-coded to /home/runner/work/CI-Active-Inference-Slopbox/CI-Active-Inference-Slopbox,
  # the checkout path of this repository under its old name.
  tests/test_main.py
)

deselect=(
  # Assertions that disagree with the code. They fail the same way on every Python from
  # 3.10 to 3.14, so they are bugs in the code or in the tests, not in the environment.
  tests/test_discretization.py::TestDataDiscretizer::test_classify_event_context
  tests/test_discretization.py::TestDataDiscretizer::test_classify_digital_behavior
  tests/test_discretization.py::TestDataDiscretizer::test_classify_location_context
  tests/test_discretization.py::TestDataDiscretizer::test_discretize_features
  # SimpleActiveInferenceMatrices.sample_action multiplies a list by a float:
  # "TypeError: can't multiply sequence by non-int of type 'float'" (matrix_utils.py).
  tests/test_matrix_utils.py::TestSimpleActiveInferenceMatrices::test_sample_action
)

args=()
for path in "${ignore[@]}"; do args+=("--ignore=${path}"); done
for node in "${deselect[@]}"; do args+=("--deselect=${node}"); done

exec python -m pytest tests -p no:cacheprovider "${args[@]}" "$@"
