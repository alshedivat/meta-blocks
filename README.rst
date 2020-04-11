
**Logo image should be adjusted**

.. image:: https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/meta-blocks-2d.png?raw=true
   :target: https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/meta-blocks-2d.png?raw=true
   :width: 200px
   :alt: Logo
   :align: center

A Modular Toolbox for Accelerating Meta-Learning Research :rocket:

----


**WARNING:** Repository is under construction. Feel free to star and subscribe for updates, but the code will be unstable and might be changing under the hood until the first beta.

.. image:: https://img.shields.io/pypi/v/meta-blocks.svg?color=brightgreen
   :target: https://pypi.org/project//meta-blocks/
   :alt: PyPI version

.. image:: https://travis-ci.org/alshedivat/meta-blocks.svg?branch=master
   :target: https://travis-ci.org/alshedivat/meta-blocks

.. image:: https://readthedocs.org/projects/meta-blocks/badge/?version=latest
   :target: https://meta-blocks.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


**Meta-Blocks** is a modular toolbox for research, experimentation, and reproducible benchmarking of learning-to-learn algorithms.
The toolbox provides flexible APIs for working with **MetaDatasets**, **TaskDistributions**, and **MetaLearners** (see the figure below).
The APIs make it easy to implement a variety of meta-learning algorithms, run them on well-established benchmarks,
or add your own meta-learning problems to the suite and benchmark algorithms on them.


.. image:: https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/system_illustration.png?raw=true
   :target: https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/system_illustration.png?raw=true
   :alt: System Illustration

**Meta-Blocks** package comes with:

* **Flexible APIs, detailed documentation, and multiple examples.**
* **Popular models and algorithms** such as MAML [#Finn2017Model]_, Reptile [#Nichol2018On]_, Protonets [#Snell2017Prototypical]_
* **Supervised and unsupervised meta-learning** setups compatible with all algorithms.
* **Customizable modules and utility functions** for quick prototyping on new meta-learning algorithms.


**Key Links and Resources**\ :

* `View the latest codes on Github <https://github.com/alshedivat/meta-blocks/>`_
* `Execute Interactive Jupyter Notebooks <https://github.com/alshedivat/meta-blocks/>`_
* `Anomaly Detection Resources <https://github.com/alshedivat/meta-blocks/>`_

----

Installation
============

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as **meta-blocks** is updated frequently:

.. code-block:: bash

   pip install meta-blocks            # normal install
   pip install --upgrade meta-blocks  # or update if needed
   pip install --pre meta-blocks      # or include pre-release version for new features

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/alshedivat/meta-blocks.git
   cd meta-blocks
   pip install .


**Required Dependencies**\ :


* albumentations
* hydra-core
* numpy
* Pillow
* scipy
* scikit-learn
* tensorflow>=2.1


----

Examples
========

TODO:
We should provide a minimal example so people could run immediately.
Ideally, the running time should be within a few mins.


----


Development
===========

For development and contributions, please make sure to install pre-commit hooks to ensure proper code style and formatting:

.. code-block:: bash

   $ pip install pre-commit      # install pre-commit
   $ pre-commit install          # install git hooks
   $ pre-commit run --all-files  # run pre-commit on all the files



Status
======

**Meta-Blocks** is currently **under development** as of Apr, 2020.

**Watch & Star** to get the latest update! Also feel free to contact for suggestions and ideas.


----


Citing Meta-Blocks
==================

TODO: add citation information as soon as available.

----


Reference
=========

.. [#Finn2017Model] Finn, C., Abbeel, P. and Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. ICML 2017.

.. [#Nichol2018On] Nichol, A., Achiam, J. and Schulman, J. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999.

.. [#Snell2017Prototypical] Snell, J., Swersky, K. and Zemel, R. Prototypical networks for few-shot learning. NeurIPS 2017.