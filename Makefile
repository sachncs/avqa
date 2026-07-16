.PHONY: help install dev test lint ruff-format format typecheck bench coverage black-check black-format clean

help:
	@echo "AVQA development targets:"
	@echo "  make install      - install package + dev deps"
	@echo "  make dev          - install with all extras"
	@echo "  make test         - run unit + integration tests"
	@echo "  make lint         - ruff check"
	@echo "  make ruff-format  - ruff format apply"
	@echo "  make black-check  - black --check (governance gate)"
	@echo "  make black-format - black format apply"
	@echo "  make format       - ruff format + black format"
	@echo "  make typecheck    - mypy strict on src/avqa/"
	@echo "  make bench        - run benchmark suite"
	@echo "  make coverage     - run tests with coverage gate (>=90%)"
	@echo "  make clean        - remove build/cache artifacts"

install:
	python -m pip install -e . --no-deps
	python -m pip install pytest pytest-cov pytest-benchmark ruff black mypy

dev:
	python -m pip install -e ".[all]" --no-deps

test:
	PYTHONPATH=src pytest tests/ -q -m "not benchmark"

lint:
	ruff check src/avqa tests/ examples/ benchmarks/ scripts/

ruff-format:
	ruff format src/avqa tests/ examples/ benchmarks/ scripts/

black-check:
	black --check src/avqa tests/ examples/ benchmarks/ scripts/

black-format:
	black src/avqa tests/ examples/ benchmarks/ scripts/

format: ruff-format black-format

typecheck:
	mypy src/avqa

bench:
	PYTHONPATH=src pytest tests/performance/ --benchmark-only --benchmark-min-rounds=2

coverage:
	PYTHONPATH=src pytest tests/unit tests/integration --cov=avqa --cov-report=term --cov-fail-under=90

clean:
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/ *.egg-info src/*.egg-info