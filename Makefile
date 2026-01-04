.PHONY: help install dev test lint format docs build clean publish-test publish

help:
	@echo "PyEval - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package"
	@echo "  make dev           Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run linting"
	@echo "  make format        Format code"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo "  make docs-serve    Serve documentation locally"
	@echo ""
	@echo "Build & Publish:"
	@echo "  make build         Build package"
	@echo "  make clean         Clean build artifacts"
	@echo "  make publish-test  Publish to TestPyPI"
	@echo "  make publish       Publish to PyPI"

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=pyeval --cov-report=html --cov-report=term-missing

lint:
	ruff check pyeval/ tests/
	mypy pyeval/ --ignore-missing-imports || true

format:
	black pyeval/ tests/
	isort pyeval/ tests/
	ruff check pyeval/ tests/ --fix

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

build: clean
	python -m build
	twine check dist/*

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf site/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*
