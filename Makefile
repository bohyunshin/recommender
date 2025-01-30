lint:
	poetry run flake8 .
	poetry run black --check .
	poetry run isort --check-only .

lint-fix:
	poetry run black .
	poetry run isort .