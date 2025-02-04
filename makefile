.PHONY: help

# Show this help.
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

lint: ## lint the code
	uv run pre-commit run --all-files

install: ## install the dependencies
	uv sync

viz: ## run kedro viz
	uv run kedro viz --autoreload

run: ## run kedro viz
	uv run kedro run
