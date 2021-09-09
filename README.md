# part-lifecycle

This project investigates the part lifecycle of high-value, long lead-time parts such as vacuum-investment cast single-crystal superalloy components with additional preprocessing steps before being ready for use.

## Problem Statement

Various industries such as aerospace and certain sectors of power generation use high-value parts that are consumed during the operational life of a machine, and require inspection and refurbishment/replacement at regular intervals. The penalty for having a machine forced out of service is extremely high, but at the same time, the parts are costly and have long lead times relative to their service life. The opportunity loss of having significant amounts of cash tied up in spare parts sitting in a warehouse must be minimized, and thus there is a clear optimization to be made. Uncertainty on lead times, quality assurance of manufactured parts (acceptance/rejection of new parts, and limited life parts), life consumption during each pass through a machine (scrap rate) introduce further complexity. I only consider a single part type, but this could be directly extended to multiple parts requiring different processing machines, processing machines of various sizes, etc. 

## Topics Covered
This analysis touches on the following topics:
- Discrete event simulation
- Monte Carlo simulation
- Global optimization
- Optimization under uncertainty, specifically using Conditional Value at Risk (CVar)
- Inventory management strategies

## Part Lifecycle

## Implementation

The part lifecycle is simulated as a discrete event simulation implemented with [SimPy](https://simpy.readthedocs.io/en/latest/). As the problem is clearly nonconvex, the cost is optimized here using global methods such as basin hopping in (scipy)[https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping] , and the methods provided by (Nevergrad)[https://facebookresearch.github.io/nevergrad/] such as particle swarm optimization, genetic algorithms, and so forth. The problem could also potentially be solved using mathematical optimization, which would bring the benefit of a more natural implementation of constraints and a lower bound on optimality via the dual problem but the scale of the problem at hand largely precludes this.

## Future Perspectives

There are many possibilities to expand this that go beyond the scope of this exploratory analysis. These include:
- Machines requiring different sizes of parts, and accordingly preprocessing machines of different sizes (and costs) are required.
- Evaluating inventory strategies when new and improved part types are introduced and old ones are phased out.
- Evaluate the potential of in-housing production to reduce lead times.

