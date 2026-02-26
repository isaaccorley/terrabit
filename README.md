# research-repo-template

A Python research repository template using [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [ty](https://github.com/astral-sh/ty), [pre-commit](https://pre-commit.com/), and GitHub Actions CI/CD.

## Features

- 📦 **[uv](https://docs.astral.sh/uv/)** — fast Python package manager and build tool
- 🔍 **[ruff](https://docs.astral.sh/ruff/)** — fast Python linter and formatter
- 🔎 **[ty](https://github.com/astral-sh/ty)** — fast Python type checker
- 🪝 **[pre-commit](https://pre-commit.com/)** — git hooks for code quality
- ✅ **GitHub Actions CI** — automated tests and style/type checks on push/PR
- 🚀 **Automatic PyPI releases** — publish to PyPI on version tag creation

## Getting Started

1. **Clone and rename** the template:

    - Replace `mypackage` with your package name throughout the repo
    - Update `pyproject.toml` with your project metadata

1. **Install dependencies** with uv:

    ```bash
    uv sync --all-groups
    ```

1. **Install pre-commit hooks**:

    ```bash
    uv run pre-commit install
    ```

1. **Run tests**:

    ```bash
    uv run pytest -vvv --cov=src
    ```

1. **Run linting and formatting**:

    ```bash
    uv run ruff check .
    uv run ruff format .
    ```

1. **Run type checking**:

    ```bash
    uv run ty check
    ```

## Releasing to PyPI

1. Set up a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for your repository with environment name `pypi`.
1. Create and push a version tag:
    ```bash
    git tag v0.1.0
    git push origin v0.1.0
    ```
    The GitHub Actions release workflow will automatically build and publish to PyPI.
