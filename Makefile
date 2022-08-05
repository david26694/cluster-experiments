black:
	black cluster_experiments tests setup.py --check

flake:
	flake8 cluster_experiments tests setup.py

test:
	pytest

check: black flake test

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"
	pre-commit install

install-test:
	python -m pip install -e ".[test]"

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*
