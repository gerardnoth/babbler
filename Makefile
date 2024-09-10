.PHONY: check format lint test typecheck

# Run all CI checks.
check : format lint typecheck test

# Format the project source code.
format:
	.venv/bin/ruff format src
	.venv/bin/ruff format tests

# Run linting on the project source.
lint:
	.venv/bin/ruff check src

# Run tests.
test:
	.venv/bin/pytest

# Type check the source code.
typecheck:
	.venv/bin/mypy src
