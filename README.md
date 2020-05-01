<p align="center"><img src="https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/meta-blocks-2d.png?raw=true" alt="logo" width="400px" /></p>

<h3 align="center">A Modular Toolbox for Accelerating Meta-Learning Research :rocket:</h3>

----

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/meta-blocks)](https://pypi.org/project/meta-blocks/)
[![PyPI Status Badge](https://badge.fury.io/py/meta-blocks.svg)](https://pypi.org/project/meta-blocks/)
[![Build Status](https://travis-ci.org/alshedivat/meta-blocks.svg)](https://travis-ci.org/alshedivat/meta-blocks)
[![Documentation Status](https://readthedocs.org/projects/meta-blocks/badge/?version=latest)](https://meta-blocks.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/alshedivat/meta-blocks/badge.svg?branch=master)](https://coveralls.io/github/alshedivat/meta-blocks?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Meta-Blocks** is a modular toolbox for research, experimentation, and reproducible benchmarking of learning-to-learn algorithms.
The toolbox provides flexible APIs for working with `MetaDatasets`, `TaskDistributions`, and `MetaLearners` (see the figure below).
The APIs make it easy to implement a variety of meta-learning algorithms, run them on well-established and emerging benchmarks, and add your own meta-learning problems to the suite and benchmark algorithms on them. 

 ![System Illustration](docs/_static/img/system_illustration.png)

**Meta-Blocks** package comes with:

* **Flexible APIs, detailed documentation, and multiple examples.**
* **Popular models and algorithms** such as MAML [1], Reptile [2], Protonets [3].
* **Supervised and unsupervised meta-learning** setups compatible with all algorithms.
* **Customizable modules and utility functions** for quick prototyping on new meta-learning algorithms.

**Links and Resources**:

* [View the latest code on Github]()
* [Execute Interactive Jupyter Notebooks]()
* [Documentation and APIs]()

---

### Installation


It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as **meta-blocks** is updated frequently:


```shell
$ pip install meta-blocks            # normal install
$ pip install --upgrade meta-blocks  # or update if needed
$ pip install --pre meta-blocks      # or include pre-release version for new features
```

Alternatively, you could clone and run setup.py file:

```
$ git clone https://github.com/alshedivat/meta-blocks.git
$ cd meta-blocks
$ pip install .
```

**Required Dependencies** :

* albumentations
* hydra-core
* numpy
* Pillow
* scipy
* scikit-learn
* tensorflow>=2.1

---

### Examples

TODO:
We should provide a minimal example so people could run immediately.
Ideally, the running time should be within a few mins.

----

### Development

For development and contributions, please install pre-commit hooks to ensure proper code style and formatting:

```shell
$ pip install pre-commit      # install pre-commit
$ pre-commit install          # install git hooks
$ pre-commit run --all-files  # run pre-commit on all the files
```

#### Status

**Meta-Blocks** is currently **under development** as of Apr, 2020.

**Watch & Star** to get the latest update! Also feel free to contact for suggestions and ideas.

----

### Citing Meta-Blocks

TODO: add citation information as soon as available.

----


### Reference

[1] Finn, C., Abbeel, P. and Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. ICML 2017.

[2] Nichol, A., Achiam, J. and Schulman, J. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999.

[3] Snell, J., Swersky, K. and Zemel, R. Prototypical networks for few-shot learning. NeurIPS 2017.
