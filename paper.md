
---
title:  'Adaptive, tools for adaptive parallel sampling of mathematical functions'
journal: 'PeerJ'
author:
- name: Tinkerer
  affiliation:
    - Kavli Institute of Nanoscience, Delft University of Technology, P.O. Box 4056, 2600 GA Delft, The Netherlands
  email: not_anton@antonakhmerov.org
abstract: |
  Large scale computer simulations are time-consuming to run and often require sweeps over input parameters to obtain a qualitative understanding of the simulation output.
  These sweeps of parameters can potentially make the simulations prohibitively expensive.
  Therefore, when evaluating a function numerically, it is advantageous to sample it more densely in the interesting regions (called adaptive sampling) instead of evaluating it on a manually-defined homogeneous grid.
  Such adaptive algorithms exist within the machine learning field.
  These mehods can suggest a new point to calculate based on \textit{all} existing data at that time; however, this is an expensive operation.
  An alternative is to use local algorithms---in contrast to the previously mentioned global algorithms---which can suggest a new point, based only on the data in the immediate vicinity of a new point.
  This approach works well, even when using hundreds of computers simultaneously because the point suggestion algorithm is cheap (fast) to evaluate.
  We provide a reference implementation in Python and show its performance.
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
<!-- This should convey the point that it is advantageous to do this. -->
A better alternative which improves the simulation efficiency is to choose new, potentially interesting points in $X$ based on existing data. [@gramacy2004parameter; @de1995adaptive; @castro2008active; @chen2017intelligent] <!-- cite i.e. hydrodynamics-->
Baysian optimization works well for high-cost simulations where one needs to find a minimum (or maximum). [@@takhtaganov2018adaptive]
If the goal of the simulation is to approximate a contineous function with the least amount of points, the continuity of the approximation is achieved by a greedy algorithm that samples mid-points of intervals with the largest Euclidean distance or curvature[@mathematica_adaptive].
Such a sampling strategy would trivially speedup many simulations.
One of the most significant complications here is to parallelize this algorithm, as it requires a lot of bookkeeping and planning ahead.

#### We describe a class of algorithms relying on local criteria for sampling, which allow for easy parallelization and have a low overhead.
Due to parallelization, the algorithm should be local, meaning that the information updates are only in a region around the newly calculated point.
Additionally, the algorithm should also be fast in order to handle many parallel workers that calculate the function and request new points.
A simple example is greedily optimizing continuity of the sampling by selecting points according to the distance to the largest gaps in the function values.
For a one-dimensional function (Fig. @fig:loss_1D) this is to (1) construct intervals containing neighboring data points, (2) calculate the Euclidean distance of each interval and assign it to the candidate point inside that interval, and finally (3) pick the candidate point with the largest Euclidean distance.
In this paper, we describe a class of algorithms that rely on local criteria for sampling, such as in the previous mentioned example.
Here we associate a *local loss* to each of the *candidate points* within an interval, and choose the points with the largest loss.
In the case of the integration algorithm the loss could just be an error estimate.
The most significant advantage of these *local* algorithms is that they allow for easy parallelization and have a low computational overhead.

![Visualization of a simple point choosing algorithm for a blackbox function (grey).
The existing data points (green) $\{x_i, y_i\}_{i \in 1...4}$ and corresponding candidate points (red) in the middle of each interval.
Each candidate point has a loss $L$ indicated by the size of the red dots.
The candidate point with the largest loss will be chosen, which in this case is the one with $L_{1,2}$.
](figures/loss_1D.pdf){#fig:loss_1D}

![Comparison of homogeneous sampling (top) with adaptive sampling (bottom) for different one-dimensional functions (red).
We see that when the function has a distince feature---such as with the peak and tanh---adaptive sampling performs much better.
When the features are homogeneously spaced, such as with the wave packet, adaptive sampling is not as effective as in the other cases.](figures/adaptive_vs_grid.pdf){#fig:adaptive_vs_grid}

#### We provide a reference implementation, the Adaptive package, and demonstrate its performance.
We provide a reference implementation, the open-source Python package called Adaptive[@Nijholt2019a], which has previously been used in several scientific publications[@vuik2018reproducing; @laeven2019enhanced; @bommer2019spin; @melo2019supercurrent].
It has algorithms for $f \colon \R^N \to \R^M$, where $N, M \in \mathbb{Z}^+$ but which work best when $N$ is small; integration in $\R$; and the averaging of stochastic functions.
Most of our algorithms allow for a customizable loss function with which one can adapt the sampling algorithm to work optimally for a specific function.
It integrates with the Jupyter notebook environment as well as popular parallel computation frameworks such as `ipyparallel`, `mpi4py`, and `dask.distributed`.
It provides auxiliary functionality such as live-plotting, inspecting the data as the calculation is in progress, and automatically saving and loading of the data.

# Review of adaptive sampling

Optimal sampling and planning based on data is a mature field with different communities providing their own context, restrictions, and algorithms to solve their problems.
To explain the relation of our approach with prior work, we discuss several existing contexts.
This is not a systematic review of all these fields, but rather, we aim to identify the important traits and design considerations.

#### Experiment design uses Bayesian sampling because the computational costs are not a limitation.
Optimal experiment design (OED) is a field of statistics that minimizes the number of experimental runs needed to estimate specific parameters, and thereby, it reduces the costs of experimentation.[@emery1998optimal]
It works with many degrees of freedom and can consider constraints, for example, when the sample space contains settings that are practically infeasible.
One form of OED is response-adaptive design[@hu2006theory], which concerns the adaptive sampling of designs for statistical experiments.
Here, the acquired data (i.e., the observations) are used to estimate the uncertainties of a certain desired parameter.
Then it suggests further experiments that will optimally reduce these uncertainties.
In this step of the calculation Bayesian statistics is frequently used.
Since Bayesian statistics naturally provides tools for answering such questions; however, because it provides closed from solutions Markov chain Monte Carlo (MCMC) sampling is the goto tool in determining the most promising samples.
In a typical non-adaptive experiment, decisions on experiments are done, are made and fixed in advance.

#### Plotting and low dimensional integration uses local sampling.
Plotting a low dimensional function in between bounds requires one to evaluate the function on sufficiently many points such that when we interpolate values in between data points, we get an accurate description of the function values that were not explicitly calculated.
In order to minimize the number of function evaluations, one can use adaptive sampling routines.
For example, for one-dimensional functions, Mathematica[@Mathematica] implements a `FunctionInterpolation` class that takes the function, $x_\textrm{min}$, and $x_\textrm{max}$, and returns an object which sampled the function in regions with high curvature more densily; however, details on the algorithm are not published.
Subsequently, we can query this object for points in between $x_\textrm{min}$ and $x_\textrm{max}$, and get the interpolated value, or we can use it to plot the function without specifying a grid.
Another application for adaptive sampling is integration.
It works by estimating integration error of each interval and then minimizing the sum of these errors greedily.
For example, the `CQUAD` algorithm[@gonnet2010increasing] in the GNU Scientific Library[@galassi1996gnu] implements a more shophisticated strategy and is a doubly-adaptive general-purpose integration routine which can handle most types of singularities.
In general, it requires more function evaluations than the integration routines in `QUADPACK`[@galassi1996gnu]; however, it works more often for difficult integrands.
It is doubly-adaptive because it can decide to either subdivide intervals into more intervals or refine an interval by adding more points---that do not lie on a regular grid---to each interval.

#### PDE solvers and computer graphics use adaptive meshing.
Hydrodynamics[@berger1989local; @berger1984adaptive] and astrophysics[@klein1999star] use adaptive refinement of the triangulation mesh at which a partial differential equation is discretized.
By providing smaller mesh elements in regions with higher variation of the function, they reduce the amount of data and calculation needed at each step of time propagation.
The remeshing at each time step happens globally and this is an expensive operation.
Therefore mesh optimization does not fit our workflow because expensive global updates should be avoided.
Computer graphics uses similar adaptive methods where a surface can be represented by a smooth surface via a coarser piecewise linear polygon mesh, called a subdivision surface[@derose1998subdivision].
An example of such a polygonal remeshing method is one where the polygons align with the curvature of the space or field, this is called anisotropic meshing[@alliez2003anisotropic].

# Design constraints and the general algorithm

#### We aim to sample low dimensional low to intermediate cost functions in parallel.
<!-- This should explain to which domain our problem belongs. -->
<!-- because of the curse of dimensionality -->
<!-- fast functions do not require adaptive -->
<!-- When your function evaluation is very expensive, full-scale Bayesian sampling will perform better; however, there is a broad class of simulations that are in the right regime for Adaptive to be beneficial. -->


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
Inspired by a method commonly employed in digital cartography for coast line simplification, we construct a loss function that does its reverse. [@visvalingam1990douglas]
<!-- https://bost.ocks.org/mike/simplify/ -->

## A parallelizable adaptive integration algorithm based on cquad

#### The `cquad` algorithm belongs to a class that is parallelizable.

## isosurface sampling
<!-- figure here -->

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