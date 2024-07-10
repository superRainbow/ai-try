# Define variables
POETRY = poetry
SRC_DIR = src
TEST_DIR = tests
LINT_TARGETS = $(SRC_DIR) $(TEST_DIR)

all: lint run clean
.PHONY: all

lint: ## Run linters
	$(POETRY) run ruff format *.py
	$(POETRY) run ruff check --fix *.py
.PHONY: lint

run: ## Run the main script
	$(POETRY) run streamlit run homepage.py
.PHONY: run

clean: ## Clean up the project
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
.PHONY: clean

# .DEFAULT_GOAL := help

help: Makefile  ## Show Makefile help
	@echo "提供以下指令使用："
	@grep -E '(^[a-zA-Z_-]+:.*?##.*$$)|(^##)' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}' | sed -e 's/\[32m##/[33m/'