from setuptools import find_packages, setup

base_packages = [
    "pip>=22.2.2",
    "statsmodels>=0.13.2",
    "pandas>=1.2.0",
    "scikit-learn>=1.0.0",
    "tqdm>=4.0.0",
    "numpy>=1.20.0",
]

only_test_packages = [
    "pytest>=5.4.3",
    "black>=19.10b0",
    "flake8>=3.8.3",
    "mktestdocs>=0.2.0",
]
test_packages = only_test_packages + base_packages

util_packages = ["pre-commit>=2.6.0", "ipykernel>=6.15.1", "twine"] + base_packages

docs_packages = [
    "mkdocs==1.2.3",
    "mkdocs-material==8.0.0",
    "mkdocstrings==0.18.0",
    "jinja2<3.1.0",
    "mkdocs-jupyter==0.22.0",
]

dev_packages = test_packages + util_packages + docs_packages

setup(
    name="cluster_experiments",
    version="0.3.2",
    packages=find_packages(),
    extras_require={
        "dev": dev_packages,
        "test": test_packages,
        "only-test": only_test_packages,
        "docs": docs_packages,
    },
    install_requires=base_packages,
)
