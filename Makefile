.PHONY: install test test-fast lint format typecheck security clean \
        docker-build docker-run pre-commit all

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v \
	  --cov=wallet_manager \
	  --cov-report=term-missing \
	  --cov-fail-under=85

test-fast:
	pytest tests/ -x -q --tb=short

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy wallet_manager/ --ignore-missing-imports

security:
	bandit -r wallet_manager/ -c pyproject.toml -ll
	pip-audit -r requirements.txt

pre-commit:
	pre-commit run --all-files

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -f coverage.xml .coverage

docker-build:
	docker build -t wallet-manager:dev .

docker-run:
	docker compose up --build

all: lint typecheck test
