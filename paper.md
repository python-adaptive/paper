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
A better alternative which improves the simulation efficiency is to choose new, potentially interesting points in $X$ based on existing data. [@gramacy2004parameter; @de1995adaptive; @castro2008active; @chen2017intelligent] <!-- cite i.e., hydrodynamics-->
Bayesian optimization works well for high-cost simulations where one needs to find a minimum (or maximum). [@@takhtaganov2018adaptive]
If the goal of the simulation is to approximate a continuous function with the least amount of points, the continuity of the approximation is achieved by a greedy algorithm that samples mid-points of intervals with the largest Euclidean distance or curvature[@mathematica_adaptive].
Such a sampling strategy would trivially speedup many simulations.
One of the most significant complications here is to parallelize this algorithm, as it requires a lot of bookkeeping and planning ahead.

![Visualization of a simple point choosing algorithm for a black box function (grey).
We start by calculating the two boundary points.
Two consecutive existing data points (black) $\{x_i, y_i\}$ define an interval.
Each interval has a loss associated with it that can be calculated from the points in the interval $L_{i,i+1}(x_i, x_{i+1}, y_i, y_{i+1})$.
At each time step the interval with the largest loss is indicated, with its corresponding candidate point (green) in the middle of the interval.
The loss function in this example is the curvature loss.
](figures/algo.pdf){#fig:algo}

#### We describe a class of algorithms relying on local criteria for sampling, which allow for easy parallelization and have a low overhead.
Due to parallelization, the algorithm should be local, meaning that the information updates are only in a region around the newly calculated point.
Additionally, the algorithm should also be fast in order to handle many parallel workers that calculate the function and request new points.
A simple example is greedily optimizing continuity of the sampling by selecting points according to the distance to the largest gaps in the function values, as in Fig. @fig:algo.
For a one-dimensional function with three points known (its boundary points and a point in the center), the following steps repeat itself:
(1) keep all points $x$ sorted, where two consecutive points define an interval,
(2) calculate the Euclidean distance for each interval (see $L_{1,2}$ in Fig. @fig:loss_1D),
(3) pick a new point $x_\textrm{new}$ in the middle of the largest interval, creating two new intervals around that point,
(4) calculate $f(x_\textrm{new})$,
(5) repeat the previous steps, without redoing calculations for unchanged intervals.
In this paper, we describe a class of algorithms that rely on local criteria for sampling, such as in the previously mentioned example.
Here we associate a *local loss* to each of the *candidate points* within an interval, and choose the points with the largest loss.
In the case of the integration algorithm, the loss is the error estimate.
The most significant advantage of these *local* algorithms is that they allow for easy parallelization and have a low computational overhead.

![Comparison of homogeneous sampling (top) with adaptive sampling (bottom) for different one-dimensional functions (red) where the number of points in each column is identical.
We see that when the function has a distinct feature---such as with the peak and tanh---adaptive sampling performs much better.
When the features are homogeneously spaced, such as with the wave packet, adaptive sampling is not as effective as in the other cases.](figures/adaptive_vs_grid.pdf){#fig:adaptive_vs_grid}

![Comparison of homogeneous sampling (top) with adaptive sampling (bottom) for different two-dimensional functions where the number of points in each column is identical.
On the left is a circle with a linear background $x + a ^ 2 / (a ^ 2 + (x - x_\textrm{offset}) ^ 2)$.
In the middle a topological phase diagram from [\onlinecite{nijholt2016orbital}] its function can be -1 or 1, which indicate the presence or abscence of a Majorana bound state.
On the right we plot level crossings for a two level quantum system.
In all cases using Adaptive results in a better plot.
](figures/adaptive_2D.pdf){#fig:adaptive_2D}


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
Since Bayesian statistics naturally provides tools for answering such questions; however, because it provides closed-form solutions Markov chain Monte Carlo (MCMC) sampling is the goto tool in determining the most promising samples.
In a typical non-adaptive experiment, decisions on experiments are done, are made and fixed in advance.

#### Plotting and low dimensional integration uses local sampling.
Plotting a low dimensional function in between bounds requires one to evaluate the function on sufficiently many points such that when we interpolate values in between data points, we get an accurate description of the function values that were not explicitly calculated.
In order to minimize the number of function evaluations, one can use adaptive sampling routines.
For example, for one-dimensional functions, Mathematica[@Mathematica] implements a `FunctionInterpolation` class that takes the function, $x_\textrm{min}$, and $x_\textrm{max}$, and returns an object which sampled the function in regions with high curvature more densely; however, details on the algorithm are not published.
Subsequently, we can query this object for points in between $x_\textrm{min}$ and $x_\textrm{max}$, and get the interpolated value, or we can use it to plot the function without specifying a grid.
Another application for adaptive sampling is integration.
It works by estimating the integration error of each interval and then minimizing the sum of these errors greedily.
For example, the `CQUAD` algorithm[@gonnet2010increasing] in the GNU Scientific Library[@galassi1996gnu] implements a more sophisticated strategy and is a doubly-adaptive general-purpose integration routine which can handle most types of singularities.
In general, it requires more function evaluations than the integration routines in `QUADPACK`[@galassi1996gnu]; however, it works more often for difficult integrands.
It is doubly-adaptive because it can decide to either subdivide intervals into more intervals or refine an interval by adding more points---that do not lie on a regular grid---to each interval.

#### PDE solvers and computer graphics use adaptive meshing.
Hydrodynamics[@berger1989local; @berger1984adaptive] and astrophysics[@klein1999star] use an adaptive refinement of the triangulation mesh at which a partial differential equation is discretized.
By providing smaller mesh elements in regions with a higher variation of the function, they reduce the amount of data and calculation needed at each step of time propagation.
The remeshing at each time step happens globally, and this is an expensive operation.
Therefore, mesh optimization does not fit our workflow because expensive global updates should be avoided.
Computer graphics uses similar adaptive methods where a smooth surface can represent a surface via a coarser piecewise linear polygon mesh, called a subdivision surface[@derose1998subdivision].
An example of such a polygonal remeshing method is one where the polygons align with the curvature of the space or field; this is called anisotropic meshing[@alliez2003anisotropic].

# Design constraints and the general algorithm

#### We aim to sample low dimensional low to intermediate cost functions in parallel.
The general algorithm that we describe in this paper works best for low to intermediary cost functions.
The point suggestion step happens in a single sequential process while the function executions can be in parallel.
This means that to benefit from an adaptive sampling algorithm $t_\textrm{function} / N_\textrm{workers} \gg t_\textrm{suggest}$ must hold, here $t_\textrm{function}$ is the average function execution time, $N_\textrm{workers}$ the number of parallel processes, and $t_\textrm{suggest}$ the time it takes to suggest a new point.
Very fast functions can be calculated on a dense grid, and extremely slow functions might benefit from full-scale Bayesian optimization where $t_\textrm{suggest}$ is large; nonetheless, a large class of functions is inside the right regime for Adaptive to be beneficial.
Further, because of the curse of dimensionality---the sparsity of space in higher dimensions---our local algorithm works best in low dimensional space; typically calculations that can reasonably be plotted, so with 1, 2, or 3 degrees of freedom.

#### We propose to use a local loss function as a criterion for choosing the next point.
To minimize $t_\textrm{suggest}$ and equivalently make the point suggestion algorithm as fast as possible, we propose to assign a loss to each interval.
This loss is determined only by the function values of the points inside that interval and optionally of its neighbouring intervals too.
The local loss function values then serve as a criterion for choosing the next point by virtue of choosing a new candidate point inside the interval with the maximum loss.
This means that upon adding new data points, only the intervals near the new point needs to have their loss value updated.
The amortized complexity of the point suggestion algorithm is, therefore, $\mathcal{O}(1)$.

#### As an example, the interpoint distance is a good loss function in one dimension.
An example of such a loss function for a one-dimensional function is the interpoint distance, such as in Fig. @fig:loss_1D.
This loss will suggest to sample a point in the middle of an interval with the largest Euclidean distance and thereby ensure the continuity of the function.
A more complex loss function that also takes the first neighbouring intervals into account is one that adds more points where the second derivative (or curvature) is the highest.
Figure @fig:adaptive_vs_grid shows a comparison between a result using this loss and a function that is sampled on a grid.

#### In general local loss functions only have a logarithmic overhead.
<!-- Bas: not sure what to write here -->

#### With many points, due to the loss being local, parallel sampling incurs no additional cost.
<!-- Bas: the text below does not really describe what is written above, but is an essential part nonetheless -->
So far, the description of the general algorithm did not include parallelism.
It needs to be able to suggest multiple points at the same time and remember which points it suggests.
When a new point $\bm{x}_\textrm{new}$ with the largest loss $L_\textrm{max}$ is suggested, the interval it belongs to splits up into $N$ new intervals (here $N$ depends on the dimensionality of the function $f$.)
A temporary loss $L_\textrm{temp} = L_\textrm{max}/N$ is assigned to these newly created intervals until $f(\bm{x})$ is calculated and the temporary loss can be replaced by the actual loss $L \equiv L((\bm{x},\bm{y})_\textrm{new}, (\bm{x},\bm{y})_\textrm{neigbors})$ of these new intervals, where $L \ge L_\textrm{temp}$.
For a one-dimensional scalar function, this procedure is equivalent to temporarily using the function values of the neighbours of $x_\textrm{new}$ and assign the interpolated value to $y_\textrm{new}$ until it is known.
When querying $n>1$ points, the above procedure simply repeats $n$ times.

# Loss function design

#### A failure mode of such algorithms is sampling only a small neighbourhood of one point.
The interpoint distance minimizing loss function we mentioned previously works on many functions; however, it is easy to write down a function where it will fail.
For example, $1/x^2$ has a singularity and will be sampled too densely around $x=0$ using this loss.
We can avoid this by defining additional logic inside the loss function.

#### A solution is to regularize the loss such that this would be avoided.
<!-- like resolution loss which limits the size of an interval -->

#### Adding loss functions allows for balancing between multiple priorities.
<!-- i.e. area + line simplification -->

#### A desirable property is that eventually, all points should be sampled.
<!-- exploration vs. exploitation -->

# Examples

## Line simplification loss

#### The line simplification loss is based on an inverse Visvalingam’s algorithm.
Inspired by a method commonly employed in digital cartography for coastline simplification, we construct a loss function that does its reverse. [@visvalingam1990douglas]
<!-- https://bost.ocks.org/mike/simplify/ -->

## A parallelizable adaptive integration algorithm based on cquad

#### The `cquad` algorithm belongs to a class that is parallelizable.

## isosurface sampling
<!-- figure here -->

# Implementation and benchmarks
We will now introduce Adaptive's API.
The object that can suggest points based on existing data is called a *learner*.
When the function value is known, we can add it to the learner's data structure, such that it can suggest more interesting points later.
We can define a learner as follows
```python
from adaptive import Learner1D

def peak(x): # pretend this is a slow function
    a = 0.01
    return x + a**2 / (a**2 + x**2)

learner = Learner1D(peak, bounds=(-1, 1))

```
To drive the learner manually (not recommended) and sequentially, we can do
```python
def goal(learner):
    # learner.loss() = max(learner.losses)
    return learner.loss() < 0.01

while not goal(learner):
    points, loss_improvements = learner.ask(n=1)
    for x in points:  # len(points) == 1
        y = f(x)
        learner.tell(x, y)
```
To do this automatically (recommended) and in parallel (by default on all cores available)
```python
from adaptive import Runner
runner = Runner(learner, goal)
```
This will return immediately because the calculation happens in the background.
That also means that as the calculation is in progress `learner.data` is accessible and plotted with `learner.plot()`.
Additionally, in a Jupyter notebook environment, one can call `runner.live_info()` to display useful information.
To change the loss function ...

```python
def default_loss(xs, ys):
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    return numpy.hypot(dx, dy)

def uniform_loss(xs, ys):
    dx = xs[1] - xs[0]
    return dx

```
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
