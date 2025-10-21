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

# References