---
title:  'Adaptive, tools for adaptive parallel sampling of mathematical functions'
journal: 'PeerJ'
author:
- name: Tinkerer
  affiliation:
    - Kavli Institute of Nanoscience, Delft University of Technology, P.O. Box 4056, 2600 GA Delft, The Netherlands
  email: not_anton@antonakhmerov.org
abstract: |
  Adaptive is an open-source Python library designed to make adaptive parallel function evaluation simple. You supply a function with its bounds and it will be evaluated at the optimal points in parameter space by analyzing existing data and planning ahead on the fly. With just a few lines of code, you can evaluate functions on a computing cluster, live-plot the data as it returns, and benefit from a significant speedup.
acknowledgements: |
  We'd like to thank ...
contribution: |
  Bla
...

# Introduction

#### Simulations are costly and often require sampling a region in parameter space.

In the computational sciences, one often does costly simulations---represented by a function $f$---where a certain region in parameter space $X$ is sampled, mapping to a codomain $Y$: $f \colon X \to Y$.
Frequently, the different points in $X$ can be independently calculated.
Even though it is suboptimal, one usually resorts to sampling $X$ on a homogeneous grid because of its simple implementation.

#### Choosing new points based on existing data improves the simulation efficiency.
A better alternative is to choose new, potentially interesting points in $X$ based on existing data, which improves the simulation efficiency.
For example, a simple strategy for a one-dimensional function is to (1) construct intervals containing neighboring data points, (2) calculate the Euclidean distance of each interval, and (3) pick the new point to be in the middle of the largest Euclidean distance.
Such a sampling strategy would trivially speedup many simulations.
One of the most significant complications here is to parallelize this algorithm, as it requires a lot of bookkeeping and planning ahead.

#### We describe a class of algorithms relying on local criteria for sampling, which allow for easy parallelization and have a low overhead.
In this paper, we describe a class of algorithms that rely on local criteria for sampling, such as in the previous simple example.
Here we associate a *local loss* with each of the *intervals* (containing neighboring points), and choose new points inside of the interval with the largest loss.
We can then easily quantify how well the data is describing the underlying function by summing all the losses; allowing us to define stopping criteria.
The most significant advantage of these algorithms is that they allow for easy parallelization and have a low computational overhead.

#### We provide a reference implementation, the Adaptive package, and demonstrate its performance.
We provide a reference implementation, the open-source Python package called Adaptive[@Nijholt2019a], which has already been used in several scientific publications[@vuik2018reproducing; @laeven2019enhanced; @bommer2019spin; @melo2019supercurrent].
It has algorithms for: $f \colon \R^N \to \R^M$, where $N, M \in \mathbb{Z}^+$ but which work best when $N$ is small; integration in $\R$; and the averaging of stoachastic functions.
Most of our algorithms allow for a customizable loss function.
In this way one can adapt the sampling algorithm to work optimally for a specific function codomain $Y$.
It easily integrates with the Jupyter notebook environment and provides tools for trivially upscaling your simulation to a computational cluster, live-plotting and inspecting the data as the calculation is in progress, automatically saving and loading of the data, and more.

# Review of adaptive sampling

#### Experiment design uses Bayesian sampling because the computational costs are not a limitation.
<!-- high dimensional functions -->

#### Plotting and low dimensional integration uses local sampling.
<!-- can refer to Mathematica's implementation -->

#### PDE solvers and computer graphics use adaptive meshing.
<!-- hydrodynamics anisotropic meshing paper ref -->

# Design constraints and the general algorithm

#### We aim to sample low dimensional low to intermediate cost functions in parallel.
<!-- because of the curse of dimensionality -->
<!-- fast functions do not require adaptive -->
<!-- When your function evaluation is very expensive, full-scale Bayesian sampling will perform better, however, there is a broad class of simulations that are in the right regime for Adaptive to be beneficial. -->

#### We propose to use a local loss function as a criterion for choosing the next point.

#### As an example interpoint distance is a good loss function in one dimension.
<!-- Plot here -->

#### In general local loss functions only have a logarithmic overhead.

#### With many points, due to the loss being local, parallel sampling incurs no additional cost.

# Loss function design

#### A failure mode of such algorithms is sampling only a small neighborhood of one point.
<!-- example of distance loss on singularities -->

#### A solution is to regularize the loss such that this would be avoided.
<!-- like resolution loss which limits the size of an interval -->

#### Adding loss functions allows for balancing between multiple priorities.
<!-- i.e. area + line simplification -->

#### A desirable property is that eventually, all points should be sampled.
<!-- exploration vs. exploitation -->

# Examples

## Line simplification loss

#### The line simplification loss is based on an inverse Visvalingam’s algorithm.
<!-- https://bost.ocks.org/mike/simplify/ -->

## A parallelizable adaptive integration algorithm based on cquad

#### The `cquad` algorithm belongs to a class that is parallelizable.

## isosurface sampling

# Implementation and benchmarks
<!-- API description -->

#### The learner abstracts a loss based priority queue.

#### The runner orchestrates the function evaluation.

# Possible extensions

#### Anisotropic triangulation would improve the algorithm.

#### Learning stochastic functions is a promising direction.

#### Experimental control needs to deal with noise, hysteresis, and the cost for changing parameters.


<!-- We can include things like:
* Asymptotically complexity of algorithms
* Setting of the problem, which classes of problems can be handled with Adaptive
* Loss-functions examples (maybe include [Adaptive quantum dots](https://chat.quantumtinkerer.tudelft.nl/chat/channels/adaptive-quantum-dots))
* Trials, statistics (such as measuring timings)
* Line simplification algorithm as a general criterium
* Desirable properties of loss-functions
* List potential applications
 -->