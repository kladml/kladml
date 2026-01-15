.PHONY: install test lint format clean build publish help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies in editable mode
	pip install -e ".[dev,tracking]"

test: ## Run tests
	pytest tests/ -v

lint: ## Run linter (ruff)
	ruff check .

format: ## Format code (ruff)
	ruff format .

clean: ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache

build: clean ## Build package
	python -m build

publish-test: build ## Publish to TestPyPI
	./scripts/publish.sh test

publish: build ## Publish to PyPI (Manual)
	./scripts/publish.sh

release: ## Create and push a new version tag (Usage: make release version=0.1.0)
	@if [ -z "$(version)" ]; then echo "❌ Error: Please specify version (e.g., make release version=0.1.0)"; exit 1; fi
	git tag -a v$(version) -m "Release v$(version)"
	git push origin v$(version)
	@echo "🚀 Release v$(version) triggered! Check GitHub Actions."
