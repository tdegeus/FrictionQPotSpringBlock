/**
\file version.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_VERSION_HPP
#define FRICTIONQPOTSPRINGBLOCK_VERSION_HPP

#include "version.h"

namespace FrictionQPotSpringBlock {

namespace detail {

inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

inline std::string version()
{
    return detail::unquote(std::string(QUOTE(FRICTIONQPOTSPRINGBLOCK_VERSION)));
}

} // namespace FrictionQPotSpringBlock

#endif
