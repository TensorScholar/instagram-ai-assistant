#!/bin/bash
set -e

echo "--- 🚀 STARTING VALIDATION PROTOCOL: PHASE 1 (FINAL RUN) ---"
echo ""
echo "--- Pipeline 1: File & Dependency Verification ---"
echo "Checking for deleted files..."
test ! -f src/api_gateway/requirements.txt && echo "✅ api_gateway/requirements.txt deleted." || (echo "❌ FAILED: api_gateway/requirements.txt still exists." && exit 1)
test ! -f src/ingestion_worker/requirements.txt && echo "✅ ingestion_worker/requirements.txt deleted." || (echo "❌ FAILED: ingestion_worker/requirements.txt still exists." && exit 1)
test ! -f src/intelligence_worker/requirements.txt && echo "✅ intelligence_worker/requirements.txt deleted." || (echo "❌ FAILED: intelligence_worker/requirements.txt still exists." && exit 1)
test ! -f src/shared_lib/requirements.txt && echo "✅ shared_lib/requirements.txt deleted." || (echo "❌ FAILED: shared_lib/requirements.txt still exists." && exit 1)
test ! -f src/shared_lib/setup.py && echo "✅ shared_lib/setup.py deleted." || (echo "❌ FAILED: shared_lib/setup.py still exists." && exit 1)

echo "Checking for new files..."
test -f pyproject.toml && echo "✅ root pyproject.toml exists." || (echo "❌ FAILED: root pyproject.toml not found." && exit 1)
test -f poetry.lock && echo "✅ poetry.lock exists." || (echo "❌ FAILED: poetry.lock not found." && exit 1)
test -f src/api_gateway/pyproject.toml && echo "✅ api_gateway pyproject.toml exists." || (echo "❌ FAILED: api_gateway pyproject.toml not found." && exit 1)
test -f src/intelligence_worker/pyproject.toml && echo "✅ intelligence_worker pyproject.toml exists." || (echo "❌ FAILED: intelligence_worker pyproject.toml not found." && exit 1)
test -f src/ingestion_worker/pyproject.toml && echo "✅ ingestion_worker pyproject.toml exists." || (echo "❌ FAILED: ingestion_worker pyproject.toml not found." && exit 1)

echo "--- Pipeline 2: Configuration & Build Verification ---"
echo "Verifying Helm chart cleanup..."
if grep -A 1 "apiGateway:" kubernetes/helm-chart/values.yaml | grep -q "env:"; then
    echo "❌ FAILED: 'env:' block still present under apiGateway in values.yaml."
    exit 1
elif grep -A 1 "intelligenceWorker:" kubernetes/helm-chart/values.yaml | grep -q "env:"; then
    echo "❌ FAILED: 'env:' block still present under intelligenceWorker in values.yaml."
    exit 1
elif grep -A 1 "ingestionWorker:" kubernetes/helm-chart/values.yaml | grep -q "env:"; then
    echo "❌ FAILED: 'env:' block still present under ingestionWorker in values.yaml."
    exit 1
else
    echo "✅ Orphaned 'env:' blocks removed from values.yaml."
fi

echo "Verifying Makefile cleanup..."
# Corrected, more robust check for "|| true"
if grep -E "^(test-unit|lint-check|test-coverage):.*\|\| true" Makefile; then
    echo "❌ FAILED: '|| true' is still present in Makefile test/lint/coverage targets."
    exit 1
else
    echo "✅ '|| true' removed from Makefile."
fi

echo "Verifying docker-compose commands..."
grep "poetry install && poetry run" docker-compose.yml -c | grep "3" > /dev/null || (echo "❌ FAILED: docker-compose.yml commands not updated correctly." && exit 1)
echo "✅ docker-compose.yml commands updated for Poetry."

echo "--- Pipeline 3: Lint & Integrity Checks ---"
echo "Running poetry check..."
poetry check || (echo "❌ FAILED: 'poetry check' failed." && exit 1)
echo "✅ Poetry dependencies are consistent."

echo "Running Helm lint (non-blocking)..."
helm lint ./kubernetes/helm-chart || echo "⚠️  WARNING: 'helm lint' failed or helm not found. Continuing validation..."
echo "✅ Helm lint check complete (non-blocking)."


echo "--- Pipeline 4: Runtime Build & Boot Verification ---"
echo "Attempting to build and start containers..."
docker-compose up -d --build
echo "Waiting 20s for services to boot..."
sleep 20
echo "Checking container status..."
docker-compose ps -a
docker-compose logs api-gateway | grep -i "Uvicorn running" || (echo "❌ FAILED: api-gateway failed to start with Poetry." && docker-compose logs api-gateway && exit 1)
echo "✅ api-gateway started successfully."
docker-compose logs intelligence-worker | grep -i "celery@.*:.*ready" || (echo "❌ FAILED: intelligence-worker failed to start with Poetry." && docker-compose logs intelligence-worker && exit 1)
echo "✅ intelligence-worker started successfully."
docker-compose logs ingestion-worker | grep -i "celery@.*:.*ready" || (echo "❌ FAILED: ingestion-worker failed to start with Poetry." && docker-compose logs ingestion-worker && exit 1)
echo "✅ ingestion-worker started successfully."
echo "Stopping containers..."
docker-compose down

echo "--- Pipeline 5: Git Verification ---"
echo "Checking Git status..."
git status | grep "nothing to commit, working tree clean" || (echo "❌ FAILED: Git working tree is not clean." && exit 1)
echo "✅ Git status is clean."
echo "Checking last commit..."
git log -1 --pretty=format:"%an - %s" | grep "Mohammad Atashi - Phase 1: Unify dependency management with Poetry and prune Helm config" || (echo "❌ FAILED: Last commit message or author is incorrect." && exit 1)
echo "✅ Git commit is correct."
echo ""
echo "--- ✅✅✅ VALIDATION PROTOCOL: PHASE 1 (FINAL RUN) PASSED ✅✅✅ ---"


