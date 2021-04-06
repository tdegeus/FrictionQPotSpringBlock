/**
Version information.

\file version.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_VERSION_H
#define FRICTIONQPOTSPRINGBLOCK_VERSION_H

#include <string>
#include <algorithm>

#include "config.h"

/**
Current version.

Either:

-   Configure using CMake at install time. Internally uses::

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using::

        -DFRICTIONQPOTSPRINGBLOCK_VERSION="`python -c "from setuptools_scm import get_version; print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using ``setuptools_scm``.
Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION``
to overwrite the automatic version.
*/
#ifndef FRICTIONQPOTSPRINGBLOCK_VERSION
#define FRICTIONQPOTSPRINGBLOCK_VERSION "@FrictionQPotSpringBlock_VERSION@"
#endif

namespace FrictionQPotSpringBlock {

/**
Return version string, e.g.::

    "0.1.0"

\return std::string
*/
inline std::string version();

} // namespace FrictionQPotSpringBlock

#include "version.hpp"

#endif
