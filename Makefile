.PHONY: help dev up down build test lint format typecheck migrate seed benchmark clean download-models logs shell

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Development ────────────────────────────────────────────

dev: ## Start all services in development mode
	docker compose -f docker-compose.yml up --build

up: ## Start all services (detached)
	docker compose -f docker-compose.yml up -d --build

down: ## Stop all services
	docker compose -f docker-compose.yml down

build: ## Build all Docker images
	docker compose -f docker-compose.yml build

logs: ## Tail logs from all services
	docker compose -f docker-compose.yml logs -f

shell: ## Open a shell in the API container
	docker compose -f docker-compose.yml exec api bash

# ─── Database ───────────────────────────────────────────────

migrate: ## Run database migrations
	docker compose -f docker-compose.yml exec api alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	docker compose -f docker-compose.yml exec api alembic revision --autogenerate -m "$(MSG)"

seed: ## Seed the database with sample data
	docker compose -f docker-compose.yml exec api python -m scripts.seed_data

# ─── Quality ────────────────────────────────────────────────

test: ## Run all tests
	docker compose -f docker-compose.yml exec api pytest tests/ -v

test-unit: ## Run unit tests only
	docker compose -f docker-compose.yml exec api pytest tests/unit/ -v

test-integration: ## Run integration tests only
	docker compose -f docker-compose.yml exec api pytest tests/integration/ -v

lint: ## Run linter (ruff)
	docker compose -f docker-compose.yml exec api ruff check cortexia/ tests/

format: ## Format code (black + ruff)
	docker compose -f docker-compose.yml exec api black cortexia/ tests/
	docker compose -f docker-compose.yml exec api ruff check --fix cortexia/ tests/

typecheck: ## Run type checker (mypy)
	docker compose -f docker-compose.yml exec api mypy cortexia/

# ─── ML Models ──────────────────────────────────────────────

download-models: ## Download ML model weights
	docker compose -f docker-compose.yml exec api python -m scripts.setup_models

# ─── Load Testing ───────────────────────────────────────────

benchmark: ## Run load tests with Locust
	docker compose -f docker-compose.yml exec api locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 60s

# ─── Cleanup ────────────────────────────────────────────────

clean: ## Remove all containers, volumes, and build artifacts
	docker compose -f docker-compose.yml down -v --rmi local
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
