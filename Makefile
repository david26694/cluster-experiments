.PHONY: clean clean-test clean-pyc clean-build

black:
	black cluster_experiments tests setup.py --check

ruff:
	ruff check cluster_experiments tests setup.py

test:
	pytest --cov=./cluster_experiments

coverage_xml:
	coverage xml

check: black ruff test coverage_xml

install:
	python -m pip install uv
	uv sync

install-dev:
	pip install --upgrade pip setuptools wheel
	python -m pip install uv
	uv sync --extra "dev"
	pre-commit install

install-dev-gh-action:
	pip install --upgrade pip setuptools wheel
	python -m pip install uv
	uv sync --extra "dev"

install-test:
	pip install --upgrade pip setuptools wheel
	python -m pip install uv
	uv sync --extra "test"

docs-deploy:
	mkdocs gh-deploy

docs-serve:
	cp README.md docs/index.md
	rm -rf docs/theme
	cp -r theme docs/theme/
	mkdocs serve

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

prep-dist: clean
	uv build

pypi: prep-dist
	twine upload --repository cluster-experiments dist/*

pypi-gh-actions: prep-dist
# todo: fix this
	twine upload --skip-existing dist/*

# Report log
report-log:
	pytest --report-log experiments/reportlog.jsonl

duration-insights:
	pytest-duration-insights explore experiments/reportlog.jsonl
