---
title: 'FastBNNs: Fast training and inference of Bayesian neural networks'
tags:
  - Python
  - PyTorch
  - Bayesian neural networks
  - variational inference
authors:
  - name: David J. Schodt
    orcid: 0000-0002-8986-2736
    affiliation: "1"
affiliations:
 - name: Los Alamos National Laboratory, USA
   index: 1

date: 8 September 2025
bibliography: paper.bib

---

# Summary
Neural networks (NNs) are a flexible class of models that can be used to approximate complicated functions.
Bayesian neural networks (BNNs) extend NNs by treating their learnable parameters as distributions, enabling uncertainty quantification of both model outputs and of the parameters themselves.
FastBNNs defines a PyTorch-based [@NEURIPS2019_9015] framework for BNN training and inference and implements a set of recently developed algorithms for fast and flexible BNN inference [@2024schodt_framework; @2024schodt_utvi].
Using thin wrappers around PyTorch modules and tensors, FastBNNs enables one-line conversion of existing neural network architectures to their Bayesian counterparts.
FastBNNs was designed to simplify and accelerate the adoption of BNNs in NN applications that benefit from uncertainty quantification.

# Statement of need
In many applications, NNs are overparameterized black-box models that can confidently produce erroneous predictions [@pmlr-v37-blundell15].
Principled approaches to uncertainty quantification are thus highly desirable for NNs, as they provide not only predictive uncertainties in model outputs but also parameter uncertainties that can drive advanced training strategies like model pruning.
BNNs treat NN parameters as distributions and hence naturally provide the desired uncertainty quantification.
Despite decades of research in BNNs, available software implementations require redefining NNs in a bespoke framework [@bingham2019pyro], are restricted to a limited set of layers [@esposito2020blitzbdl; @krishnan2022bayesiantorch], and/or do not model heteroscedastic uncertainty in the data [@laplace2021].

FastBNNs provides a Python implementation of a set of recently developed algorithms for approximate Bayesian inference [@2024schodt_framework; @2024schodt_utvi] that support variational inference of BNNs assuming a mean-field approximation (i.e., independent parameters).
By wrapping existing PyTorch-based models, FastBNNs enables fast and flexible BNN training and inference without sacrificing the simplicity, flexibility, and familiarity of PyTorch.
FastBNNs fills a gap in existing open-source BNN software by simplifying NN-to-BNN conversion and leveraging fast inference algorithms, encouraging further research in and adoption of BNNs.

# Comparison to similar software
To demonstrate the utility of FastBNNs, we compare predictive performance and approximate runtime for a simple regression problem across a selection of open-source BNN inference software.
For our comparison, we model data sampled from the function $y = ax + cx^3 + \epsilon (x)$ where $\epsilon (x) \sim \mathcal{N}(\mu=0, \sigma^2=0.1 + 0.2 \cos{(2 \pi x)}^2)$ is heteroscedastic noise.
We train a 2-layer multilayer perceptron (MLP) with 8 hidden units and a custom, learnable nonlinearity defined by $h(\tilde{x}) = A \tilde{x}^3$, where $\tilde{x} \in \mathbb{R}^8$ is the input feature and $A \in \mathbb{R}^{8 \times 8}$ is learned.
Training samples are generated for inputs $x \sim \mathcal{U}[-0.5, 0.5]$, while test samples are generated for inputs $x \sim \mathcal{U}[-1.0, 1.0]$ to compare out-of-distribution (O.O.D.) performance to in-distribution performance (I.D.).

We compare FastBNNs to two open-source software packages for BNN inference: Laplace [@laplace2021], which implements several variants of the Laplace approximation, of which we use the diagonal (mean-field) Laplace approximation; and Bayesian-Torch [@krishnan2022bayesiantorch], which implements mean-field Monte Carlo-based inference for common NN layers.
Visualizations of both I.D. and O.O.D. test-set predictions for each of the trained models are shown in \autoref{fig:comparison}, as well as approximate inference times as measured on an NVIDIA RTX 2000 Ada Generation Laptop GPU.
The script used to train and evaluate these models is included in the FastBNNs repository at [polynomial.py](https://github.com/lanl/FastBNNs/blob/main/comparisons/polynomial.py).

Notably, FastBNNs is faster than sampling-based inference as used in Bayesian-Torch, and for this example and hardware combination, is even faster than the generalized linear model inference used in laplace-torcch.
Furthermore, although FastBNNs and Bayesian-Torch (and not Laplace) can model heteroscedastic aleatoric uncertainty, Bayesian-Torch is not able to model the parameter distribution of the custom nonlinearity, which may explain the inaccurate uncertainty predictions for the I.D. test set evaluations seen in \autoref{fig:comparison}.

![Comparison between FastBNNs, Bayesian-Torch, and laplace-torch. Test-set evaluations are made for I.D. and O.O.D. data for models trained using FastBNNs, Bayesian-Torch (with 30 Monte Carlo samples), and laplace-torch (using the diagonal Laplace approximation). Inference times are shown for an NVIDIA RTX 2000 Ada Generation Laptop GPU.\label{fig:comparison}](../comparisons/polynomial.png)


# References