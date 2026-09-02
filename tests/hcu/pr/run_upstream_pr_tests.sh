#!/usr/bin/env bash
set -euo pipefail

# The default branch pull_request_target workflow still invokes this path while
# the repository-owned test-only workflow change is under review.
echo "Upstream PR tests are disabled; repository-owned tests run in run_hcu_unit_tests.sh."
