<p align="center"><img src="https://github.com/alshedivat/meta-blocks/blob/master/docs/_static/img/meta-blocks-2d.png?raw=true" alt="logo" width="400px" /></p>

<h3 align="center">A Modular Toolbox for Accelerating Meta-Learning Research :rocket:</h3>

---

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/meta-blocks)](https://pypi.org/project/meta-blocks/)
[![PyPI Status Badge](https://badge.fury.io/py/meta-blocks.svg)](https://pypi.org/project/meta-blocks/)
[![Build Status](https://travis-ci.org/alshedivat/meta-blocks.svg?branch=master)](https://travis-ci.org/alshedivat/meta-blocks)
[![Documentation Status](https://readthedocs.org/projects/meta-blocks/badge/?version=latest)](https://meta-blocks.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/alshedivat/meta-blocks/badge.svg?branch=master)](https://coveralls.io/github/alshedivat/meta-blocks?branch=master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/alshedivat/meta-blocks.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/alshedivat/meta-blocks/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**MetaBlocks** is a modular toolbox for research, experimentation, and reproducible benchmarking of learning-to-learn algorithms.
The toolbox provides a flexible API for working with `MetaDatasets`, `TaskDistributions`, and `MetaLearners`.
The API fully decouples data processing, construction of learning tasks (and their distributions), adaptation algorithms, and model architectures, and makes it easy to experiment with different combinations of these basic building blocks as well as add new components to the growing ecosystem.
Additionally, the library features a [suite of benchmarks](benchmarks) that enable reproducibility.
Everything is highly configurable through [hydra](https://hydra.cc/).

The library is **under active development**.
The latest documentation is available at: https://meta-blocks.readthedocs.io/.

---

## Installation

MetaBlocks requires Python 3.5+ and TensorFlow 2.2+.

### For typical use

We recommend using **pip** for installing the latest release of the library:
```shell script
$ pip install meta-blocks            # normal install
$ pip install --upgrade meta-blocks  # or update if needed
$ pip install --pre meta-blocks      # or include pre-release version for new features
```

Alternatively, to install the latest version from the `master` branch:
```shell script
$ git clone https://github.com/alshedivat/meta-blocks.git
$ pip install meta-blocks
```

**Note:** to be able to access and run benchmarks, you will need to clone the repository.

### For development and contributions

You can install additional development requirements as follows:
```shell script
$ pip install -r requirements/dev.txt
```

Also, please make sure to install pre-commit hooks to ensure proper code style and formatting:
```shell script
$ pip install pre-commit      # install pre-commit
$ pre-commit install          # install git hooks
$ pre-commit run --all-files  # run pre-commit on all the files
```

---

## Getting started & use cases

You can use the library as (1) a modular benchmarking suite or (2) a scaffold API for new learning-to-learn algorithms.

### Benchmarking

To enable reproducible research, we maintain a suite of [benchmarks/](benchmarks/).
To run a benchmark, simply clone the repo, change your working directory to the corresponding benchmark, and execute a run script.
For example:
```shell script
$ git clone https://github.com/alshedivat/meta-blocks.git
$ cd meta-blocks/benchmarks/omniglot
$ ./fetch_data                    # fetches data for the benchmark
$ ./run_classic_supervised.sh     # runs an experiment (train and eval routines in parallel)
```
For more details, please refer to [benchmarks/README.md](benchmarks/README.md).

### MetaBlocks API

**MetaBlocks** provides multiple layers of API implemented as a hierarchy of Python classes.
The three major main components are `MetaDataset`, `TaskDistribution`, and `MetaLearner`:

1. `MetaDataset` provides access to `Dataset` instances constructed from an underlying `DataSource`.
   `Dataset` represents a collection of data tensors (e.g., in case of multi-class classification, it is a collection of input tensors, one for each class).
2. `TaskDistribution` further builds on top of `MetaDataset` and provides access to `Task` instances that specify the semantics of a learning task.
   E.g., a few-shot classification task provides access to non-overlapping support and query subsets of a `Dataset`.
   Task distributions determine how tasks are sampled and constructed.
   Currently, we support `supervised` and `self-supervised` tasks for few-shot classification.
3. `MetaLearner` encapsulates a parametric model (your favorite neural net) and an adaptation algorithm used for adapting the model to new tasks.
   Adaptation algorithms must use the API exposed by the `Task`.

**Note:** decoupling tasks from datasets and (meta-)learning methods is one of the core advantages of meta-blocks over other libraries.
   
Below are the components currently supported by the library:

<table>
<thead>
<th>Component</th>
<th colspan="4">Supported Instances</th>
</thead>
<tr>
<td>MetaDataset</td>
<td>Omniglot</td>
<td>MiniImageNet</td>
<td colspan="2">...</td>
</tr>
<tr>
<td>TaskDistribution</td>
<td>Classic supervised</td>
<td>Limited supervised</td>
<td>Self-supervised</td>
<td>...</td>
</tr>
<tr>
<td>MetaLearner</td>
<td>MAML [1]</td>
<td>Reptile [2]</td>
<td>Prototypical Networks [3]</td>
<td>...</td>
</tr>
</table>

#### Adding your own meta-datasets

To add your own meta-datasets, you need to subclass `MetaDataset` and implement a few methods.

[TODO: provide a detailed walk-through example.]

If the full data used to construct the meta-dataset is light and easily fits in the memory, you can follow [the implementation of Omniglot](meta_blocks/datasets/omniglot.py).
If the dataset is too large or requires some heavy preprocessing, the best way is to use `tf.data.Dataset` API.
As a starting point, you can follow the [miniImageNet implementation](meta_blocks/datasets/miniimagenet.py).


#### Adding your own meta-learners

To add your own meta-learning algorithms, you need to subclass `MetaLearner` and implement two methods: `_get_adapted_model` (must return an adapted model instance) and `_build_adaptation` (must build a part of the computation graph that adapts the model).
Example: [prototype-based adaptation](meta_blocks/adaptation/proto.py) builds prototypes from the support set inside `_build_adaptation` method and returns a model with the corresponding prototypes when `_get_adapted_model` is called.

[TODO: provide a detailed walk-through example.]

---

## Citation

If you use meta-blocks for research, please cite it as follows.

```
@misc{metablocks,
  title={MetaBlocks: A modular toolbox for meta-learning research with a focus on speed and reproducibility.},
  year={2020},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/alshedivat/meta-blocks}},
}
```

---

## Related projects

A few notable related projects:

| Project | Description |
| ------- | ----------- |
| [Torchmeta](https://github.com/tristandeleu/pytorch-meta) | A PyTorch library that implements multiple few-shot learning methods. |
| [learn2learn](https://github.com/learnables/learn2learn)  | A PyTorch library that supports meta-RL. |


## References

[1] Finn, C., Abbeel, P. and Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. ICML 2017.

[2] Nichol, A., Achiam, J. and Schulman, J. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999.

[3] Snell, J., Swersky, K. and Zemel, R. Prototypical networks for few-shot learning. NeurIPS 2017.
