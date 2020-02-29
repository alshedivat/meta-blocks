from setuptools import setup, find_packages

# read the contents of README file
from os import path
from io import open

# get __version__ from _version.py
ver_file = path.join("meta_blocks", "version.py")
with open(ver_file) as fp:
    exec(fp.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    # readme_path = path.join(this_directory, "README.rst")
    readme_path = path.join(this_directory, "README.md")
    with open(readme_path, encoding="utf-8") as fp:
        return fp.read()


# read the contents of requirements.txt
def requirements():
    requirements_path = path.join(this_directory, "requirements.txt")
    with open(requirements_path, encoding="utf-8") as fp:
        return fp.read().splitlines()


setup(
    name="meta-blocks",
    version=__version__,
    description="A modular toolbox for accelerating meta-learning research :rocket:",
    long_description=readme(),
    # long_description_content_type="text/x-rst",
    long_description_content_type='text/markdown',
    url="https://github.com/alshedivat/meta-blocks",
    author="Maruan Al-Shedivat, Yue Zhao",
    author_email="alshedivat@cs.cmu.edu, zhaoy@cmu.edu",
    license="BSD-3",
    keywords=[
         "deep learning",
         "machine learning",
         "explainability",
         "interpretability",
         "tensorflow",
         "keras",
         "python",
     ],
     packages=find_packages(exclude=["tests"]),
     package_data={
         "meta_blocks": [
             "configs/*.yaml",
             "configs/**/*.yaml",
             "configs/**/**/*.yaml",
             "configs/**/**/**/*.yaml",
             "configs/**/**/**/**/*.yaml",
         ],
     },
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
         "Programming Language :: Python :: 3.5",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
     ],
)
