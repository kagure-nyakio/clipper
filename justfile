default:
    @just --list
    
# Install/sync all dependencies (ruff, mypy, deptry, pytest, pytest-cov)
install:
    uv sync --frozen

format: install
    uv run ruff format --check .

lint: install
    uv run ruff check .

typecheck: install
    uv run mypy src/

deps-check: install
    uv run deptry src

lock-check: install
    uv lock --check

test: install
    uv run pytest --cov=src --cov-report=term-missing

ci: format lint typecheck deps-check lock-check test
