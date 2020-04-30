from io import open
from os import path

from setuptools import find_packages, setup

# Get __version__ from _version.py.
__version__ = None
ver_file = path.join("meta_blocks", "version.py")
with open(ver_file) as fp:
    exec(fp.read())

this_directory = path.abspath(path.dirname(__file__))


# Load README.
def readme():
    # readme_path = path.join(this_directory, "README.rst")
    readme_path = path.join(this_directory, "README.md")
    with open(readme_path, encoding="utf-8") as fp:
        return fp.read()


# Load requirements.
def requirements():
    requirements_path = path.join(this_directory, "requirements/base.txt")
    with open(requirements_path, encoding="utf-8") as fp:
        return fp.read().splitlines()


setup(
    name="meta_blocks",
    version=__version__,
    description="A modular toolbox for accelerating meta-learning research :rocket:",
    long_description=readme(),
    # long_description_content_type="text/x-rst",
    long_description_content_type="text/markdown",
    url="https://github.com/alshedivat/meta-blocks",
    author="Maruan Al-Shedivat",
    author_email="alshedivat@cs.cmu.edu",
    maintainer="Maruan Al-Shedivat, Yue Zhao",
    maintainer_email="alshedivat@cs.cmu.edu, zhaoy@cmu.edu",
    license="BSD-3",
    keywords=[
        "learning-to-learn",
        "machine learning",
        "deep learning",
        "meta-learning",
        "benchmark",
        "toolbox",
        "tensorflow",
        "keras",
        "python",
    ],
    packages=find_packages(include=["meta_blocks", "hydra_plugins.meta_blocks"]),
    include_package_data=True,
    install_requires=requirements(),
    setup_requires=["setuptools>=38.6.0"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
