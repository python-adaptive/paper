# Adaptive

We propose to use local criteriums for sampling combined with global updates.
This defines a family of straightforwardly parallelizable algorithms which are useful for intermediary cost simulations.

When your function evaluation is very expensive, full-scale Bayesian sampling will perform better, however, there is a broad class of simulations that are in the right regime for Adaptive to be beneficial.

We can include things like:
* Asymptotically complexity of algorithms
* Setting of the problem, which classes of problems can be handled with Adaptive
* Loss-functions examples (maybe include [Adaptive quantum dots](https://chat.quantumtinkerer.tudelft.nl/chat/channels/adaptive-quantum-dots))
* Trials, statistics (such as measuring timings)
* Line simplification algorithm as a general criterium
* Desirable properties of loss-functions
* List potential applications
