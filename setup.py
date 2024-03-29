from pathlib import Path

from setuptools_scm import get_version
from skbuild import setup

project_name = "FrictionQPotSpringBlock"
description = "Spring-block friction model with local disordered potential energy landscape."

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=project_name,
    description=description,
    long_description=long_description,
    version=get_version(),
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    url=f"https://github.com/tdegeus/{project_name}",
    packages=[f"{project_name}"],
    package_dir={"": "python"},
    cmake_install_dir=f"python/{project_name}",
    cmake_minimum_required_version="3.13",
    python_requires=">=3.6",
)
