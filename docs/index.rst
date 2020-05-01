Welcome to meta-blocks's documentation!
=======================================

**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/pyversions/meta-blocks
   :target: https://pypi.org/project/meta-blocks/
   :alt: PyPI - Python Version

.. image:: https://badge.fury.io/py/meta-blocks.svg
   :target: https://pypi.org/project/meta-blocks/
   :alt: PyPI Version

.. image:: https://travis-ci.org/alshedivat/meta-blocks.svg
   :target: https://travis-ci.org/alshedivat/meta-blocks
   :alt: Build Status

**Meta-Blocks** is a modular toolbox for research, experimentation, and reproducible benchmarking of learning-to-learn algorithms.
The toolbox provides flexible APIs for working with ``MetaDatasets``, ``TaskDistributions``, and ``MetaLearners`` (see the figure below).
The APIs make it easy to implement a variety of meta-learning algorithms, run them on well-established benchmarks,
or add your own meta-learning problems to the suite and benchmark algorithms on them.

.. image:: _static/img/system_illustration.png
   :alt: System Illustration

----

**Meta-Blocks** package comes with:

* **Flexible APIs, detailed documentation, and multiple examples.**
* **Popular models and algorithms** such as MAML :cite:`finn2017model`, Reptile :cite:`nichol2018first`, Protonets :cite:`snell2017prototypical`.
* **Supervised and unsupervised meta-learning** setups compatible with all algorithms.
* **Customizable modules and utility functions** for quick prototyping on new meta-learning algorithms.

**Key Links and Resources:**

* `View the latest codes on Github <https://github.com/alshedivat/meta-blocks/>`_
* `Execute Interactive Jupyter Notebooks <https://github.com/alshedivat/meta-blocks/>`_
* `Anomaly Detection Resources <https://github.com/alshedivat/meta-blocks/>`_

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   examples

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   meta_blocks

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   about
   faq
   whats_new

----

.. rubric:: References

.. bibliography:: references.bib
   :cited:

----

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
