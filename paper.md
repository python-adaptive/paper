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

#### Simulations are costly and often require sampling a region in parameter space

#### Chosing new points based on existing data improves the simulation efficiency
<!-- examples here -->

#### We describe a class of algorithms replying on local criteria for sampling which allow for easy parallelization and have a low overhead
<!-- This is useful for intermediary cost simulations. -->

#### We provide a reference implementation, the Adaptive package, and demonstrate its performance

# Review of adaptive sampling

#### Experiment design uses Bayesian sampling because the computational costs are not a limitation
<!-- high dimensional functions -->

#### Plotting and low dimensional integration uses local sampling
<!-- can refer to Mathematica's implementation -->

#### PDE solvers and computer graphics use adaptive meshing
<!-- hydrodynamics anisotropic meshing paper ref -->

# Design constraints and the general algorithm

#### We aim to sample low dimensional low to intermediate cost functions in parallel
<!-- because of curse of dimensionality -->
<!-- fast functions don't require adaptive -->
<!-- When your function evaluation is very expensive, full-scale Bayesian sampling will perform better, however, there is a broad class of simulations that are in the right regime for Adaptive to be beneficial. -->

#### We propose to use a local loss function as a criterion for chosing the next point

#### As an example interpoint distance is a good loss function in one dimension
<!-- Plot here -->

#### In general local loss functions only have a logarithmic overhead

#### With many points, due to the loss being local, parallel sampling incurs no additional cost

# Loss function design

#### A failure mode of such algorithms is sampling only a small neighborhood of one point
<!-- example of distance loss on singularities -->

#### A solution is to regularize the loss such that this would avoided
<!-- like resolution loss which limits the size of an interval -->

#### Adding loss functions allows for balancing between multiple priorities
<!-- i.e. area + line simplification -->

#### A desireble property is that eventually all points should be sampled
<!-- exploration vs. explotation -->

# Examples

## Line simplification loss

#### The line simplification loss is based on an inverse Visvalingam’s algorithm
<!-- https://bost.ocks.org/mike/simplify/ -->

## A parallelizable adaptive integration algorithm based on cquad

#### The `cquad` algorithm belongs to a class that is parallelizable

## isosurface sampling

# Implementation and benchmarks
<!-- API description -->

#### The learner abstracts a loss based priority queue

#### The runner orchestrates the function evaluation

# Possible extensions

#### Anisotropic triangulation would improve the algorithm

#### Learning stochastic functions is promising direction

#### Experimental control needs to deal with noise, hysteresis, and the cost for changing parameters


<!-- We can include things like:
* Asymptotically complexity of algorithms
* Setting of the problem, which classes of problems can be handled with Adaptive
* Loss-functions examples (maybe include [Adaptive quantum dots](https://chat.quantumtinkerer.tudelft.nl/chat/channels/adaptive-quantum-dots))
* Trials, statistics (such as measuring timings)
* Line simplification algorithm as a general criterium
* Desirable properties of loss-functions
* List potential applications
 -->