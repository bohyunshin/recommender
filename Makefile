lint:
	uv run ruff format .
	uv run ruff check .

test:
	uv run pytest

build-ext:
	uv run python setup.py build_ext --inplace
