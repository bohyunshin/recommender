lint:
	poetry run flake8 .
# 	poetry run pylint --recursive=y .
# 	poetry run black --check .
# 	poetry run isort --check-only .

lint-fix:
	poetry run black .
	poetry run isort .