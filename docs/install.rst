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


**Required Dependencies:**

* albumentations
* hydra-core
* numpy
* Pillow
* scipy
* scikit-learn
* tensorflow==2.2.0rc3
