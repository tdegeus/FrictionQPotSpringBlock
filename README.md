# FrictionQPotSpringBlock

[![CI](https://github.com/tdegeus/FrictionQPotSpringBlock/workflows/CI/badge.svg)](https://github.com/tdegeus/FrictionQPotSpringBlock/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/FrictionQPotSpringBlock/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/FrictionQPotSpringBlock)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/frictionqpotspringblock.svg)](https://anaconda.org/conda-forge/frictionqpotspringblock)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/python-frictionqpotspringblock.svg)](https://anaconda.org/conda-forge/python-frictionqpotspringblock)

Spring-block friction model with local disordered potential energy landscape.

# Change-log

## v0.6.5

*   Adding time (#33)
*   Allowing for extra iterations

## v0.6.4

*   Using simple stop criterion in event driven read (#32)

## v0.6.3

*   Run a series of time-steps until the next plastic event (#31)

## v0.6.2

*   Switching to xtensor-python (no API change).

## v0.6.1

*   Fixing bug in triggering.

## v0.6.0

*   Updating Python API. Minor style updates. (#26)
*   Fixing bug in event driven code. (#26)
*   Adding advanceElastic.
*   Adding debug assertions.
*   Renaming "advanceRightElastic" -> "advanceEventRightElastic" (#24)

## v0.5.0

*   Adding function to trigger weakest.
*   Adding function to time avalanche.
*   Recomputing forces after quench.
*   Adding Python test (#21).
*   Updating readme.

## v0.4.0

*   Switch to QPot/Chunked and prrng. Drawing the energy landscape is now done manually externally, but it allows much easier reproducibility and customisation.

## v0.3.3

*   Wrapping some QPot::RedrawList function for Python API that has a mysterious bug using the reference (#16)

## v0.3.2

*   Reconstruction using new QPot features (#14)

## v0.3.0

*   Adding redraw customization (#12)

## v0.2.0

*   Updating to new QPot
*   Various minor updates
*   Using doxygen docs (publishing GH Action)
*   Recover configuration to make random simulations deterministic: C++ and Python example
*   Updating appropriate forces at the appropriate instances
*   Added output "Load" example for future checking
*   Adding Python API 
*   Using setuptools_scm for versioning
*   Improving CI
