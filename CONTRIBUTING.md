# Contributing to KladML

Thank you for your interest in contributing to KladML! We welcome contributions from the community to help make this project better.

## Code of Conduct

Please adhere to our Code of Conduct in all interactions with the community.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/kladml.git
    cd kladml
    ```
3.  **Set up the development environment**:
    ```bash
    # Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # on Windows: .venv\Scripts\activate

    # Install dev dependencies
    pip install -e ".[dev]"
    ```

## Development Workflow

1.  **Create a branch** for your feature or bugfix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Make your changes**. Use the KladML coding style (Standard Python recommendations, `black` formatting).
3.  **Run tests** to ensure no regressions:
    ```bash
    pytest
    ```
4.  **Lint your code**:
    ```bash
    ruff check .
    ```

## Submitting Changes

1.  **Commit your changes** with descriptive commit messages.
    - We follow [Conventional Commits](https://www.conventionalcommits.org/).
    - Example: `feat: add new timeseries evaluator` or `fix: resolve issue with loading configuration`.
2.  **Push to your fork**:
    ```bash
    git push origin feature/my-new-feature
    ```
3.  **Open a Pull Request** (PR) on the main repository.
    - Provide a clear title and description.
    - Link to any relevant issues.

## Reporting Bugs

If you find a bug, please open an issue on GitHub including:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, KladML version)

## Feature Requests

We welcome new ideas! Open an issue with the "enhancement" label to discuss your proposal before starting implementation.
