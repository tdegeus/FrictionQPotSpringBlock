# FrictionQPotSpringBlock

[![CI](https://github.com/tdegeus/FrictionQPotSpringBlock/actions/workflows/ci.yml/badge.svg)](https://github.com/tdegeus/FrictionQPotSpringBlock/actions/workflows/ci.yml)
[![Doxygen -> gh-pages](https://github.com/tdegeus/FrictionQPotSpringBlock/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/FrictionQPotSpringBlock)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/frictionqpotspringblock.svg)](https://anaconda.org/conda-forge/frictionqpotspringblock)

Spring-block friction model with local disordered potential energy landscape.

# Python module

## Using conda

```bash
conda install -c conda-forge python-frictionqpotspringblock
```

Note that *xsimd* and hardware optimisations are **not enabled**.
To enable them you have to compile on your system, as is discussed next.

## From source

>   You need *xtensor*, *xtensor-python* and optionally *xsimd* as prerequisites.
>   Additionally, Python needs to know how to find them.
>   The easiest is to use *conda* to get the prerequisites:
>
>   ```bash
>   conda install -c conda-forge xtensor-python
>   conda install -c conda-forge xsimd
>   ```
>
>   If you then compile and install with the same environment
>   you should be good to go.
>   Otherwise, a bit of manual labour might be needed to
>   treat the dependencies.

```bash
# Download FrictionQPotSpringBlock
git checkout https://github.com/tdegeus/FrictionQPotSpringBlock.git
cd FrictionQPotSpringBlock

# Only if you want to use hardware optimization:
export CMAKE_ARGS="-DUSE_SIMD=1"

# Compile and install the Python module
# (-vv can be omitted as is controls just the verbosity)
python -m pip install . -vv
```

# Change-log

## v0.16.0

### API Changes

*   [BREAKING CHANGE] `minimise_nopassing`: unifying return
*   [BREAKING CHANGE] `minimise`: changing the return value
*   [BREAKING CHANGE] `timeStepsUntilEvent`: no throw if convergence was not reached
*   [BREAKING CHANGE] `timeSteps`: adding `nmargin` option (note that the default of 1 implies the breaking change)
*   [BREAKING CHANGE] [Python] To guide the user: use functions instead of properties when a computation is involved
*   [BREAKING CHANGE] Removing: `yieldDistanceRight` / `yieldDistanceLeft`. Adding: `y_right` / `y_left`
*   [BREAKING CHANGE] Removing: `triggerWeakest`

### Bug Fixes

*   `set_t`: ensure proper round

### Internal Changes

*   Internal code-style updates
*   Fixing typo in CMakeLists
*   [docs] Readme: fixing minor typos
*   Merging "version.h" in "config.h"
*   Adding version test

## v0.13.0

### API Changes

*   Python API: converting many functions to properties
*   Pre-processor switch FRICTIONQPOTSPRINGBLOCK_USE_XTENSOR_PYTHON to using xtensor-python instead of xtensor
*   Remove unused class for maintainability

## Internal changes

*   Fixing compiling warnings
*   Upgrading to Catch2.v3
*   Remove MSC work-around

## v0.12.0

### API changes

*   [BREAKING CHANGE] The constructor overloads with default `istart = zeros` are removed
    to enhance maintainability.
    See [#51](https://github.com/tdegeus/FrictionQPotSpringBlock/pull/51).

*   [BREAKING CHANGE] Allowe the use of `minimise_timeactivity` with chunked storage.
    See [#53](https://github.com/tdegeus/FrictionQPotSpringBlock/pull/53).

*   Adding function to advance to a fixed force.
    See [#52](https://github.com/tdegeus/FrictionQPotSpringBlock/pull/52).

## v0.11.0

*   Adding `SystemThermalRandomForcing`: thermal system under external forcing
    (i.e. constant force, without a 'driving frame').

## v0.10.1

### Internal changes

*   Changing `xt::xtensor` -> `xt::pytensor`.
    An alias `xt::pytensor == xt::xtensor` is created for compiling without xtensor-python.

## v0.10.0

### Changes

*   [BREAKING CHANGE] Integrating `boundscheck` in API of `minimise`.

## v0.9.0

### Bugfixes & extensions

*   Fixing ``nopassing`` algorithm for overdamped dynamics.
*   Increasing the default maximum number of iterations.
*   Adding `ymax` overload.

### Internal

*   Optimization: precompute `1 / m`.

### Documentation

*   Chuncked example: avoid costly search.

## v0.8.0

### Bugfixes & extensions

*   Calculating instantaneous temperature.
*   Temporarily work around xtensor#2510.
*   Adding SystemThermalRandomForcing.

## v0.7.0

## Bugfixes & extensions

*   Resolving segmentation fault windows.
*   Updating deps
*   [BREAKING CHANGE] Making `advanceElastic` private
*   Adding `minimise_nopassing` with fully overdamped dynamics
*   [BREAKING CHANGE] Renaming `y` -> `refChunked`. Adding array interfaces
*   [BREAKING CHANGE] Cleaning API of separate functions to set parameters
*   Fixing minor typos.
*   [Python] Adding missing imports

### Documentation

*   Simplifying example.
*   Removing matplotlib from test runs.
*   Simplifying example (historic data `txt` -> `h5`, does not lead to change, this was checked).
*   Updating badges in readme.
*   Adding full Python example.

## v0.6.8

*   [Python] Using scikit-build.

## v0.6.7

*   Avoiding setuptools_scm dependency if SETUPTOOLS_SCM_PRETEND_VERSION is defined

## v0.6.6

*   Python: passing `CMAKE_ARGS` environment variable.

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
