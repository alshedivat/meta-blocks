.. meta-blocks documentation master file, created by
   sphinx-quickstart on Wed Mar  4 20:14:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to meta-blocks's documentation!
=======================================


**Deployment & Documentation & Stats**

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

As shown in the Figure below, it contains xxx modules, and each of them can be tailored to ...

.. image:: https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/system_illustration.png?raw=true
   :target: https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/system_illustration.png?raw=true
   :alt: System Illustration

**Meta-Blocks** package comes with:

* **Flexible APIs, detailed documentation, and multiple examples.**
* **Popular models and algorithms** such as MAML :cite:`finn2017model`, Reptile :cite:`nichol2018first`, Protonets :cite:`snell2017prototypical`
* **Supervised and unsupervised meta-learning** setups compatible with all algorithms.
* **Customizable modules and utility functions** for quick prototyping on new meta-learning algorithms.


**Key Links and Resources**\ :

* `View the latest codes on Github <https://github.com/alshedivat/meta-blocks/>`_
* `Execute Interactive Jupyter Notebooks <https://github.com/alshedivat/meta-blocks/>`_
* `Anomaly Detection Resources <https://github.com/alshedivat/meta-blocks/>`_


----


.. toctree::
   :maxdepth: 2
   :caption: Contents: Getting Started

   install
   example


.. toctree::
   :maxdepth: 2
   :caption: Contents: Documentation

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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
