.PHONY: help setup install dev test lint format typecheck metadata dist-check bench coverage clean

help:
	@echo "AVQA development targets:"
	@echo "  make setup        - create .venv and install all development extras"
	@echo "  make install      - install package + dev deps"
	@echo "  make dev          - install with dev and visualization extras"
	@echo "  make test         - run unit + integration tests"
	@echo "  make lint         - ruff check"
	@echo "  make format       - ruff format apply"
	@echo "  make typecheck    - mypy strict on src/avqa/"
	@echo "  make metadata     - validate synchronized release metadata"
	@echo "  make dist-check   - validate and smoke-test built artifacts"
	@echo "  make bench        - run benchmark suite"
	@echo "  make coverage     - run tests with coverage gate (>=90%)"
	@echo "  make clean        - remove build/cache artifacts"

setup:
	./setup.sh

install:
	python -m pip install -e ".[dev]"

dev:
	python -m pip install -e ".[dev,viz]"

test:
	PYTHONPATH=src pytest tests/ -q -m "not benchmark"

lint:
	ruff check src/avqa tests/ examples/ benchmarks/ scripts/

format:
	ruff format src/avqa tests/ examples/ benchmarks/ scripts/

typecheck:
	mypy src/avqa

metadata:
	python scripts/check_metadata.py

dist-check:
	./scripts/check_dist.sh

bench:
	PYTHONPATH=src pytest tests/performance/ --benchmark-only --benchmark-min-rounds=2

coverage:
	PYTHONPATH=src pytest tests/unit tests/integration --cov=avqa --cov-report=term --cov-fail-under=90

clean:
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/ *.egg-info src/*.egg-info
