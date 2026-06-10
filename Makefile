.PHONY: fix test run run-api run-dashboard install

install:
	uv sync --all-extras

fix:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

run: run-api

run-api:
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	uv run streamlit run dashboard/app.py --server.port 8501
