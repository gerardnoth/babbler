Babbler
=======

A CLI app for interacting with generative APIs.

Development
-----------

These instructions will get you started with developing the project.

### Prerequisites

The following are required to develop this project:

- Python 3.12 and associated development packages.
- [Poetry](https://python-poetry.org): a Python package and dependency manager.

#### Python

Install packages for Python development.

```bash
sudo apt update
sudo apt install python3.12 python3.12-dev python3.12-venv
```

You may need to add the [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) PPA to resolve the packages:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

#### Poetry

Follow the installation instructions given in the [official docs](https://python-poetry.org/docs/). A brief reference is included here.

Install on Linux:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Configure Poetry to create virtual environments in the project root:

```bash
poetry config virtualenvs.in-project true --local
```

Other local configurations can be made by modifying [poetry.toml](poetry.toml). See [_Local configuration_](poetry.toml).

Now in the project root, you can install the project dependencies with Poetry:

```bash
poetry install
```

This will install dependencies into a virtual environment located in the `.venv` directory. Commands can be run in the virtual environment by using `poetry run` or by activating the virtual environment.

Dependency Management
---------------------

Dependencies can be added or removed by using Poetry. See [_Specifying dependencies_
](https://python-poetry.org/docs/basic-usage/#specifying-dependencies) and [_Installing dependencies_](https://python-poetry.org/docs/basic-usage/#installing-dependencies).

The following example uses the [loguru](https://github.com/Delgan/loguru) package to demonstrate how to use Poetry.

Add a dependency to `pyproject.toml`:

```bash
poetry add loguru
```

Remove a dependency from `pyproject.toml`:

```bash
poetry remove loguru
```

The dependencies specified in `pyproject.toml` can be edited directly instead of using Poetry. The `poetry.lock` file must then be manually regenerated and installed to the virtual environment after editing it.

Add or remove a dependency under `tool.poetry.dependencies`:

```toml
[tool.poetry.dependencies]
loguru = "0.7.2"
```

Regenerate lock file without updating locked versions:

```bash
poetry lock --no-update
```

Install dependencies specified in the lock file:

```bash
poetry install
```

Testing
-------

The project has tests for correctness and style. The full suite can be run with the following command:

```bash
make check
```

### Formatting and Linting

[Ruff](https://docs.astral.sh/ruff/) is used for formatting and linting source code. Ruff can be configured by modifying the `[tool.ruff]` sections in [pyproject.toml](pyproject.toml). See [_Configuring Ruff_](https://docs.astral.sh/ruff/configuration/).

Format code:

```bash
make format
```

Lint code:

```bash
make lint
```

Linting errors can be ignored by adding `# noqa: <error>` to the end of a line.

### Type Checking

[Mypy](https://mypy-lang.org/) is used for type checking. Type checking can be configured by modifying the `[tool.mypy]` section in [pyproject.toml](pyproject.toml).

Run type checking:

```bash
make typecheck
```

Type errors can be ignored by adding `# type: ignore` to the end of a line.

### Tests

[Pytest](https://pytest.org/) is used for running tests. Tests are located within the `tests` directory. Pytest can be configured under the `[tool.pytest.ini_options]` section in [pyproject.toml](pyproject.toml). See [_Configuration_](https://docs.pytest.org/en/stable/reference/customize.html) for more options.

Run tests:

```bash
make tests
```

IDE
---

### PyCharm

[PyCharm](https://www.jetbrains.com/pycharm/) may need to be configured before it correctly detects the project. Right-click on `src`, then under `Mark directory as`, select `Sources Root`. Restart the IDE and it should work. If it doesn't, follow the steps under `Main Menu > File > Repair IDE`.
