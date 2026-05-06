# NaiveGMM : An unsupervised clustering method for the efficient separation of two overlapping Gaussians

NaiveGMM is an unsupervised clustering method specifically designed to separate two overlapping n-dimensional Gaussian distributions. For this specific, yet wide-ranging task, NaiveGMM outperforms more generalized state-of-the art clustering techniques. The mathematical formalism behind this new method is described in details in a paper currently under review.

This repository contains the method itself, and a Jupyter notebook to test NaiveGMM's capabilities on challenging real world applications and simulated datasets, and compare with those of K-Means or AutoGMM.

To run NaiveGMM, you will need basic Python libraries (numpy, copy, warnings, itertools, optionally matplotlib) and scipy.

To run the notebook, you will also need Jupyter, Scikit-Learn (for K-means, AutoGMM and the Wisconsin Breast Cancer dataset), and Pandas (to read the Wisconsin Breast Cancer dataset).



Author: A. Picquenot, 2026



