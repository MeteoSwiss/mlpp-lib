# minimal setup.y until poetry supports PEP660
# https://github.com/python-poetry/poetry/issues/34
from setuptools import setup, find_packages

requirements = [
    "numpy",
    "tensorflow",
    "tensorflow-probability"
]

setup(
    name="mlpp-lib",
    install_requires=requirements,
    packages=find_packages(include=["mlpp_lib"]),
)
