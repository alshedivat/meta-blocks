<p align="center"><img src="https://github.com/alshedivat/meta-blocks/blob/master/docs/figs/meta-blocks-2d.png?raw=true" alt="logo" width="400px" /></p>

<h3 align="center">A Modular Toolbox for Accelerating Meta-Learning Research :rocket:</h3>

----

**This section is reserved for badges. To add**:

PyPI Badge | Travis CI Badge | Circle CI Badge | docs Badge | Coverage Badge


**Meta-Blocks** is a modular toolbox for research, experimentation, and reproducible benchmarking of learning-to-learn algorithms.
The toolbox provides flexible APIs for working with `MetaDatasets`, `TaskDistributions`, and `MetaLearners` (see the figure below).
The APIs make it easy to implement a variety of meta-learning algorithms, run them on well-established benchmarks, or add your own meta-learning problems to the suite and benchmark algorithms on them. 

 ![System Illustration](docs/figs/system_illustration.png)

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
   pip install meta-blocks            # normal install
   pip install --upgrade meta-blocks  # or update if needed
   pip install --pre meta-blocks      # or include pre-release version for new features
```

Alternatively, you could clone and run setup.py file:

```
   git clone https://github.com/alshedivat/meta-blocks.git
   cd meta-blocks
   pip install .
```

**Required Dependencies**\ :

* albumentations
* hydra-core
* numpy
* Pillow
* scipy
* scikit-learn
* tensorflow>=2.1


---

### Example

We should provide a minimal example so people could run immediately. Ideally the running time should be within a few mins.


----


### Development Status


**Meta-Blocks** is currently **under development** as of Feb, 2020.

**Watch & Star** to get the latest update! Also feel free to contact for suggestions and ideas.

----

### Citing Meta-Blocks

**Meta-Blocks** paper can be accessed [here]().
If you use Meta-Blocks in a scientific publication, we would appreciate citations to the following paper (to be fixed):

```
    @inproceedings{zhao2020combo,
      title={Combining Machine Learning Models and Scores using combo library},
      author={Zhao, Yue and Wang, Xuejian and Cheng, Cheng and Ding, Xueying},
      booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},
      month = {Feb},
      year={2020},
      address = {New York, USA}
    }
```

or

```
    Zhao, Y., Wang, X., Cheng, C. and Ding, X., 2020. Combining Machine Learning Models and Scores using combo library. Thirty-Fourth AAAI Conference on Artificial Intelligence.
```

----


### Reference

[1] Finn, C., Abbeel, P. and Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. ICML 2017.

[2] Nichol, A., Achiam, J. and Schulman, J. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999.

[3] Snell, J., Swersky, K. and Zemel, R. Prototypical networks for few-shot learning. NeurIPS 2017.
