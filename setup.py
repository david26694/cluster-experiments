from setuptools import find_packages, setup

base_packages = ["pandas>=1.0.0"]

test_packages = [
    "pytest>=5.4.3",
    "black>=19.10b0",
    "flake8>=3.8.3",
] + base_packages

util_packages = ["pre-commit>=2.6.0"]

docs_packages = [
    "mkdocs>=1.1",
    "mkdocs-material>=4.6.3",
    "mkdocstrings>=0.8.0",
]

dev_packages = test_packages + util_packages + docs_packages

setup(
    name="cluster_experiments",
    version="0.1.0",
    packages=find_packages(),
    extras_require={
        "dev": dev_packages,
        "test": test_packages,
    },
)
