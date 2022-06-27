/**
Version information.

\file version.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_VERSION_H
#define FRICTIONQPOTSPRINGBLOCK_VERSION_H

#include <algorithm>
#include <string>

#include "config.h"

/**
Current version.

Either:

-   Configure using CMake at install time. Internally uses::

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using::

        -DFRICTIONQPOTSPRINGBLOCK_VERSION="`python -c "from setuptools_scm import get_version;
print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
overwrite the automatic version.
*/
#ifndef FRICTIONQPOTSPRINGBLOCK_VERSION
#define FRICTIONQPOTSPRINGBLOCK_VERSION "@PROJECT_VERSION@"
#endif

namespace FrictionQPotSpringBlock {

namespace detail {

inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

/**
Return version string, e.g.::

    "0.1.0"

\return std::string
*/
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(FRICTIONQPOTSPRINGBLOCK_VERSION)));
}

} // namespace FrictionQPotSpringBlock

#endif
