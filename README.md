# Quantum Machine Learning Experiments
This is a course project for ECE594BB (Selected Topics in High-Dimensional Tensor Data Analysis) instructed by Dr. Zheng Zhang at University of California, Santa Barbara. This project aims to utilize the [Google TensorNetwork toolbox](https://github.com/google/TensorNetwork) to perform machine learning over the tree tensor network inspired by quantum computing. This project is mainly based on the following work:

> [1] Ding Liu, Shi-Ju Ran, Peter Wittek, Cheng Peng, Raul Blázquez García, Gang Su, and Maciej Lewenstein. [Machine learning by unitary tensor network of hierarchical tree structure](https://iopscience.iop.org/article/10.1088/1367-2630/ab31ef), *New Journal of Physics*, 21(7), 073059, 2019.

Their code is also available on [Github link](https://github.com/dingliu0305/Tree-Tensor-Networks-in-Machine-Learning).

## Requirements

The code is Python3-based, and the following packages are required to run the repository.

- [TensorNetwork](https://github.com/google/TensorNetwork)
- [tncontract](https://github.com/andrewdarmawan/tncontract)
- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch](https://pytorch.org/)

## Installation

1. Download the entire repository
2. Copy corresponding data files from this [link](https://ucsb.box.com/s/u6hx1wt7pdqab7ojye7gsep9ij44g39d), and the put the content into a folder called `data`.

## Basic Examples

To reproduce the results from [1], the following command can be executed.

`./qmle reproduce --data-path ./data`

To run the experiments using TensorNetwork, the following command can be executed.

`./qmle run --data-path ./data --dataset MNIST`

## Arguments

Please check the arguements of the program with

`./qmle reproduce -h`

or

`./qmle run -h`

There are mainly a few groups of arguments.

- Arguments for logging and output directory: `prefix`, `log`, `log-level`
- Arguments for input: `data-path`, `dataset`
- Arguments for paper reproducing hyperparameter: `num-epoch`, `bond-data`, `bond-inner`, `num-train-single`, `num-test-each`

For more details, please check `utils/arg_parse.py` file.

## TODO
1. Test different hyperparameter setting for the paper reproducing, and collect the results (Jose Acuna)
2. Try more sophiticated normalization method for the image, please check function `image_normalization` in `./third-party/ttn_ref.py`. (Jose Acuna)
3. Use TensorNetwork to implement the entire flow, and generalize for more datasets (Hanbin Hu)
