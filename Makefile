# Aura Platform Makefile
# Comprehensive development and deployment commands

.PHONY: help dev-up dev-down logs test lint build clean health-check k8s-deploy

# Default target
help: ## Show this help message
	@echo "Aura Platform - Available Commands:"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# DEVELOPMENT ENVIRONMENT
# =============================================================================
dev-up: ## Start all services for development
	@echo "🚀 Starting Aura Platform development environment..."
	docker-compose up -d
	@echo "⏳ Waiting for services to be healthy..."
	@$(MAKE) health-check
	@echo "✅ Development environment is ready!"

dev-down: ## Stop all services
	@echo "🛑 Stopping Aura Platform services..."
	docker-compose down
	@echo "✅ All services stopped"

dev-restart: ## Restart all services
	@echo "🔄 Restarting Aura Platform services..."
	docker-compose restart
	@echo "✅ All services restarted"

dev-logs: ## View logs from all services
	docker-compose logs -f

dev-logs-api: ## View API Gateway logs
	docker-compose logs -f api-gateway

dev-logs-intelligence: ## View Intelligence Worker logs
	docker-compose logs -f intelligence-worker

dev-logs-ingestion: ## View Ingestion Worker logs
	docker-compose logs -f ingestion-worker

# =============================================================================
# HEALTH CHECKS
# =============================================================================
health-check: ## Check health of all services
	@echo "🔍 Checking service health..."
	@echo "PostgreSQL: $$(docker-compose exec -T postgres pg_isready -U aura_user -d aura_platform > /dev/null 2>&1 && echo '✅ Healthy' || echo '❌ Unhealthy')"
	@echo "RabbitMQ: $$(docker-compose exec -T rabbitmq rabbitmq-diagnostics ping > /dev/null 2>&1 && echo '✅ Healthy' || echo '❌ Unhealthy')"
	@echo "Milvus: $$(docker-compose exec -T milvus curl -f http://localhost:9091/healthz > /dev/null 2>&1 && echo '✅ Healthy' || echo '❌ Unhealthy')"
	@echo "Vault: $$(docker-compose exec -T vault vault status > /dev/null 2>&1 && echo '✅ Healthy' || echo '❌ Unhealthy')"
	@echo "ETCD: $$(docker-compose exec -T etcd etcdctl endpoint health > /dev/null 2>&1 && echo '✅ Healthy' || echo '❌ Unhealthy')"
	@echo "MinIO: $$(docker-compose exec -T minio curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1 && echo '✅ Healthy' || echo '❌ Unhealthy')"

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	@echo "🧪 Running all tests..."
	@$(MAKE) test-unit
	@$(MAKE) test-integration

test-unit: ## Run unit tests
	@echo "🧪 Running unit tests..."
	@for service in api_gateway intelligence_worker ingestion_worker shared_lib; do \
		echo "Testing $$service..."; \
		docker-compose exec -T $$service poetry run pytest tests/ -v; \
	done

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	@if [ -f "tests/test_e2e_flow.py" ]; then \
		python tests/test_e2e_flow.py; \
	else \
		echo "⚠️  Integration tests not yet implemented for Phase 0"; \
	fi

test-coverage: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	@for service in api_gateway intelligence_worker ingestion_worker shared_lib; do \
		echo "Testing $$service with coverage..."; \
		docker-compose exec -T $$service poetry run pytest tests/ --cov=app --cov-report=html; \
	done

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linting and formatting
	@echo "🔍 Running code quality checks..."
	@$(MAKE) lint-check
	@$(MAKE) lint-format

lint-check: ## Run linting (flake8)
	@echo "🔍 Running flake8 linting..."
	@for service in api_gateway intelligence_worker ingestion_worker shared_lib; do \
		echo "Linting $$service..."; \
		docker-compose exec -T $$service poetry run flake8 app/; \
	done

lint-format: ## Run code formatting (black)
	@echo "🎨 Running black formatting..."
	@for service in api_gateway intelligence_worker ingestion_worker shared_lib; do \
		echo "Formatting $$service..."; \
		docker-compose exec -T $$service poetry run black app/; \
	done

lint-fix: ## Fix linting issues automatically
	@echo "🔧 Fixing linting issues..."
	@$(MAKE) lint-format
	@echo "✅ Linting issues fixed"

# =============================================================================
# BUILDING
# =============================================================================
build: ## Build all Docker images
	@echo "🔨 Building Docker images..."
	docker-compose build
	@echo "✅ All images built successfully"

build-no-cache: ## Build all Docker images without cache
	@echo "🔨 Building Docker images (no cache)..."
	docker-compose build --no-cache
	@echo "✅ All images built successfully"

# =============================================================================
# CLEANUP
# =============================================================================
clean: ## Clean up containers and volumes
	@echo "🧹 Cleaning up..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup completed"

clean-volumes: ## Remove all volumes (WARNING: This will delete all data)
	@echo "⚠️  WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ]
	docker-compose down -v
	docker volume prune -f
	@echo "✅ All volumes removed"

clean-images: ## Remove all Docker images
	@echo "🧹 Removing Docker images..."
	docker-compose down
	docker image prune -f
	@echo "✅ Images cleaned"

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================
db-migrate: ## Run database migrations
	@echo "🗄️  Running database migrations..."
	docker-compose exec postgres psql -U aura_user -d aura_platform -f /docker-entrypoint-initdb.d/init-db.sql
	@echo "✅ Database migrations completed"

db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "⚠️  WARNING: This will delete all database data!"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ]
	docker-compose down
	docker volume rm aura-platform_postgres_data
	docker-compose up -d postgres
	@echo "✅ Database reset completed"

db-shell: ## Connect to database shell
	docker-compose exec postgres psql -U aura_user -d aura_platform

# =============================================================================
# KUBERNETES DEPLOYMENT
# =============================================================================
k8s-deploy: ## Deploy to Kubernetes using Helm
	@echo "🚀 Deploying to Kubernetes..."
	helm install aura ./kubernetes/helm-chart
	@echo "✅ Deployment completed"

k8s-upgrade: ## Upgrade Kubernetes deployment
	@echo "🔄 Upgrading Kubernetes deployment..."
	helm upgrade aura ./kubernetes/helm-chart
	@echo "✅ Upgrade completed"

k8s-uninstall: ## Uninstall Kubernetes deployment
	@echo "🗑️  Uninstalling Kubernetes deployment..."
	helm uninstall aura
	@echo "✅ Uninstall completed"

k8s-status: ## Check Kubernetes deployment status
	@echo "📊 Kubernetes deployment status:"
	kubectl get pods -l app.kubernetes.io/name=aura
	kubectl get services -l app.kubernetes.io/name=aura

# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================
shell-api: ## Open shell in API Gateway container
	docker-compose exec api-gateway /bin/bash

shell-intelligence: ## Open shell in Intelligence Worker container
	docker-compose exec intelligence-worker /bin/bash

shell-ingestion: ## Open shell in Ingestion Worker container
	docker-compose exec ingestion-worker /bin/bash

shell-postgres: ## Open shell in PostgreSQL container
	docker-compose exec postgres /bin/bash

shell-rabbitmq: ## Open shell in RabbitMQ container
	docker-compose exec rabbitmq /bin/bash

# =============================================================================
# MONITORING
# =============================================================================
monitor: ## Show resource usage
	@echo "📊 Resource usage:"
	docker stats --no-stream

logs-tail: ## Tail logs from all services
	docker-compose logs -f --tail=100

# =============================================================================
# SECRETS MANAGEMENT
# =============================================================================
vault-init: ## Initialize Vault
	@echo "🔐 Initializing Vault..."
	docker-compose exec vault vault auth -method=userpass username=admin password=admin
	@echo "✅ Vault initialized"

vault-status: ## Check Vault status
	docker-compose exec vault vault status

# =============================================================================
# PHASE-SPECIFIC COMMANDS
# =============================================================================
phase0-setup: ## Complete Phase 0 setup
	@echo "🏗️  Setting up Phase 0..."
	@$(MAKE) dev-up
	@$(MAKE) health-check
	@echo "✅ Phase 0 setup completed"

phase0-validate: ## Validate Phase 0 setup
	@echo "🔍 Validating Phase 0 setup..."
	@$(MAKE) health-check
	@echo "✅ Phase 0 validation completed"

# =============================================================================
# INFORMATION
# =============================================================================
info: ## Show system information
	@echo "📋 Aura Platform Information:"
	@echo "=============================="
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "Make version: $$(make --version | head -n1)"
	@echo "Python version: $$(python3 --version 2>/dev/null || echo 'Python not found')"
	@echo "Current directory: $$(pwd)"
	@echo "Environment: $$(echo $$ENVIRONMENT || echo 'Not set')"

version: ## Show version information
	@echo "Aura Platform v0.1.0 - Phase 0 Foundation"
	@echo "Built with Docker, Kubernetes, and Python 3.11+"
