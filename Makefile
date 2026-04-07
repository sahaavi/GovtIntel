.PHONY: install test lint format run clean

install:
	pip install -e ".[dev]"

test:
	pytest -v

test-cov:
	pytest -v --cov=govintel --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

run:
	uvicorn govintel.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf .mypy_cache/ htmlcov/ .coverage
