# CorneaForge Makefile
#
# USAGE:
#   make install     Install package + dev deps via uv
#   make lint        Check code style
#   make format      Auto-fix code style
#   make test        Run test suite
#   make check       lint + test (what CI does)
#   make serve       Dev server on :8000
#   make docker      Build Docker image
#   make clean       Remove build artifacts

.PHONY: install install-server lint format test check serve docker clean

install:
	uv sync --extra dev

install-server:
	uv sync --extra dev --extra server

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest -q --tb=short

check: lint test

serve:
	uvicorn corneaforge.server:app --reload --host 0.0.0.0 --port 8000

docker:
	docker build -t corneaforge .

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
