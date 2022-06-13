/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_HPP
#define FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_HPP

#include "Line1d.h"

namespace FrictionQPotSpringBlock {
namespace Line1d {

inline std::vector<std::string> version_dependencies()
{
    std::vector<std::string> ret;

    ret.push_back("frictionqpotspringblock=" + version());
    ret.push_back("goosefem=" + GooseFEM::version());
    ret.push_back("qpot=" + QPot::version());

    ret.push_back(
        "xtensor=" + detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

#ifdef XSIMD_VERSION_MAJOR
    ret.push_back(
        "xsimd=" + detail::unquote(std::string(QUOTE(XSIMD_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_PATCH))));
#endif

#ifdef XTL_VERSION_MAJOR
    ret.push_back(
        "xtl=" + detail::unquote(std::string(QUOTE(XTL_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_PATCH))));
#endif

#if defined(XTENSOR_PYTHON_VERSION_MAJOR)
    ret.push_back(
        "xtensor-python=" + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MAJOR))) +
        "." + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_PATCH))));
#endif

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

    return ret;
}

template <class T, class I>
inline System::System(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y,
    const I& istart)
{
    this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, x_y, istart);
}

template <class T, class I>
inline void System::initSystem(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y,
    const I& istart)
{
#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 2);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(istart.dimension() == 1);
    auto y0 = xt::view(x_y, xt::all(), 0);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(y0 < 0.0));
#endif

    m_N = x_y.shape(0);
    m_m = m;
    m_inv_m = 1.0 / m;
    m_eta = eta;
    m_mu = mu;
    m_k_neighbours = k_neighbours;
    m_k_frame = k_frame;
    m_dt = dt;
    m_f = xt::zeros<double>({m_N});
    m_f_potential = xt::zeros<double>({m_N});
    m_f_neighbours = xt::zeros<double>({m_N});
    m_f_frame = xt::zeros<double>({m_N});
    m_f_damping = xt::zeros<double>({m_N});
    m_x = xt::zeros<double>({m_N});
    m_v = xt::zeros<double>({m_N});
    m_a = xt::zeros<double>({m_N});
    m_v_n = xt::zeros<double>({m_N});
    m_a_n = xt::zeros<double>({m_N});
    m_y.resize(m_N);

    for (size_t p = 0; p < m_N; ++p) {
        m_y[p] = QPot::Chunked(m_x(p), xt::eval(xt::view(x_y, p, xt::all())), istart(p));
    }

    this->updated_x();
    this->updated_v();
}

inline size_t System::N() const
{
    return m_N;
}

inline array_type::tensor<double, 2> System::y()
{
    size_t n = static_cast<size_t>(m_y[0].cend() - m_y[0].cbegin());
    array_type::tensor<double, 2> ret = xt::empty<double>({m_N, n});

    for (size_t p = 0; p < m_N; ++p) {
        size_t m = static_cast<size_t>(m_y[p].cend() - m_y[p].cbegin());
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(m == n);
        std::copy(m_y[p].cbegin(), m_y[p].cend(), &ret(p, xt::missing));
    }

    return ret;
}

inline QPot::Chunked& System::refChunked(size_t p)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
    return m_y[p];
}

template <class I, class T>
inline void System::set_y(const I& istart, const T& x_y)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(istart, {m_N}));
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 2);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.shape(0) == m_N);

    for (size_t p = 0; p < m_N; ++p) {
        m_y[p].set_y(istart(p), xt::eval(xt::view(x_y, p, xt::all())));
    }
}

template <class I, class T>
inline void System::shift_y(const I& istart, const T& x_y, size_t nbuffer)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(istart, {m_N}));
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 2);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.shape(0) == m_N);

    for (size_t p = 0; p < m_N; ++p) {
        m_y[p].shift_y(istart(p), xt::eval(xt::view(x_y, p, xt::all())), nbuffer);
    }
}

template <class I, class T>
inline void System::shift_dy(const I& istart, const T& dx_y, size_t nbuffer)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(istart, {m_N}));
    FRICTIONQPOTSPRINGBLOCK_ASSERT(dx_y.dimension() == 2);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(dx_y.shape(0) == m_N);
    size_t n = xt::strides(dx_y, 0);

    for (size_t p = 0; p < m_N; ++p) {
        m_y[p].shift_dy(istart(p), &dx_y(p, xt::missing), &dx_y(p, xt::missing) + n, nbuffer);
    }
}

template <class T>
inline void System::set_y(size_t p, long istart, const T& x_y)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 1);
    m_y[p].set_y(istart, x_y);
}

template <class T>
inline void System::shift_y(size_t p, long istart, const T& x_y, size_t nbuffer)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 1);
    m_y[p].shift_y(istart, x_y, nbuffer);
}

template <class T>
inline void System::shift_dy(size_t p, long istart, const T& dx_y, size_t nbuffer)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(dx_y.dimension() == 1);
    m_y[p].shift_dy(istart, dx_y, nbuffer);
}

inline array_type::tensor<double, 1> System::ymin() const
{
    array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].ymin();
    }

    return ret;
}

inline array_type::tensor<double, 1> System::ymax() const
{
    array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].ymax();
    }

    return ret;
}

inline array_type::tensor<double, 1> System::ymin_chunk() const
{
    array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].ymin_chunk();
    }

    return ret;
}

inline array_type::tensor<double, 1> System::yleft() const
{
    array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].yleft();
    }

    return ret;
}

inline array_type::tensor<double, 1> System::yright() const
{
    array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].yright();
    }

    return ret;
}

inline array_type::tensor<size_t, 1> System::i_chunk() const
{
    array_type::tensor<size_t, 1> ret = xt::empty<size_t>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].i_chunk();
    }

    return ret;
}

inline array_type::tensor<long, 1> System::istart() const
{
    array_type::tensor<long, 1> ret = xt::empty<long>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].istart();
    }

    return ret;
}

inline array_type::tensor<long, 1> System::istop() const
{
    array_type::tensor<long, 1> ret = xt::empty<long>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].istop();
    }

    return ret;
}

inline array_type::tensor<bool, 1> System::inbounds_left(size_t n) const
{
    array_type::tensor<bool, 1> ret = xt::empty<bool>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].inbounds_left(n);
    }

    return ret;
}

inline array_type::tensor<bool, 1> System::inbounds_right(size_t n) const
{
    array_type::tensor<bool, 1> ret = xt::empty<bool>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].inbounds_right(n);
    }

    return ret;
}

inline bool System::all_inbounds_left(size_t n) const
{
    for (size_t p = 0; p < m_N; ++p) {
        if (!m_y[p].inbounds_left(n)) {
            return false;
        }
    }

    return true;
}

inline bool System::all_inbounds_right(size_t n) const
{
    for (size_t p = 0; p < m_N; ++p) {
        if (!m_y[p].inbounds_right(n)) {
            return false;
        }
    }

    return true;
}

inline bool System::all_inbounds(size_t n) const
{
    for (size_t p = 0; p < m_N; ++p) {
        if (!m_y[p].inbounds(n)) {
            return false;
        }
    }

    return true;
}

inline bool System::any_redraw() const
{
    for (size_t p = 0; p < m_N; ++p) {
        auto r = m_y[p].redraw(m_x(p));
        if (r != 0) {
            return true;
        }
    }

    return false;
}

template <class T>
inline bool System::any_redraw(const T& xtrial) const
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(xtrial, {m_N}));

    for (size_t p = 0; p < m_N; ++p) {
        auto r = m_y[p].redraw(xtrial(p));
        if (r != 0) {
            return true;
        }
    }

    return false;
}

inline array_type::tensor<double, 1> System::yieldDistanceRight() const
{
    return this->yright() - m_x;
}

inline array_type::tensor<double, 1> System::yieldDistanceLeft() const
{
    return m_x - this->yleft();
}

inline array_type::tensor<long, 1> System::i() const
{
    array_type::tensor<long, 1> ret = xt::empty<long>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].i();
    }

    return ret;
}

inline void System::set_t(double arg)
{
    m_inc = static_cast<size_t>(arg / m_dt);
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::allclose(this->t(), arg));
}

inline void System::set_inc(size_t arg)
{
    m_inc = arg;
}

inline void System::set_x_frame(double arg)
{
    m_x_frame = arg;
    this->computeForceFrame();
    this->computeForce();
}

template <class T>
inline void System::set_x(const T& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
    xt::noalias(m_x) = arg;
    this->updated_x();
}

template <class T>
inline void System::set_v(const T& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
    xt::noalias(m_v) = arg;
    this->updated_v();
}

template <class T>
inline void System::set_a(const T& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
    xt::noalias(m_a) = arg;
}

inline double System::t() const
{
    return static_cast<double>(m_inc) * m_dt;
}

inline size_t System::inc() const
{
    return m_inc;
}

inline double System::temperature() const
{
    return xt::norm_sq(m_v)() * m_m / m_N;
}

inline double System::x_frame() const
{
    return m_x_frame;
}

inline array_type::tensor<double, 1> System::x() const
{
    return m_x;
}

inline array_type::tensor<double, 1> System::v() const
{
    return m_v;
}

inline array_type::tensor<double, 1> System::a() const
{
    return m_a;
}

inline array_type::tensor<double, 1> System::f() const
{
    return m_f;
}

inline array_type::tensor<double, 1> System::f_potential() const
{
    return m_f_potential;
}

inline array_type::tensor<double, 1> System::f_frame() const
{
    return m_f_frame;
}

inline array_type::tensor<double, 1> System::f_neighbours() const
{
    return m_f_neighbours;
}

inline array_type::tensor<double, 1> System::f_damping() const
{
    return m_f_damping;
}

inline void System::updated_x()
{
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForce();
}

inline void System::updated_v()
{
    this->computeForceDamping();
    this->computeForce();
}

inline void System::computeForcePotential()
{
    for (size_t p = 0; p < m_N; ++p) {
        double x = m_x(p);
        m_y[p].set_x(x);
        m_f_potential(p) = m_mu * (0.5 * (m_y[p].yright() + m_y[p].yleft()) - x);
    }
}

inline void System::computeForceNeighbours()
{
    for (size_t p = 1; p < m_N - 1; ++p) {
        m_f_neighbours(p) = m_k_neighbours * (m_x(p - 1) - 2 * m_x(p) + m_x(p + 1));
    }
    m_f_neighbours.front() = m_k_neighbours * (m_x.back() - 2 * m_x.front() + m_x(1));
    m_f_neighbours.back() = m_k_neighbours * (m_x(m_N - 2) - 2 * m_x.back() + m_x.front());
}

inline void System::computeForceFrame()
{
    xt::noalias(m_f_frame) = m_k_frame * (m_x_frame - m_x);
}

inline void System::computeForceDamping()
{
    xt::noalias(m_f_damping) = -m_eta * m_v;
}

inline void System::computeForce()
{
    xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame;
}

inline double System::residual() const
{
    double r_fres = xt::norm_l2(m_f)();
    double r_fext = xt::norm_l2(m_f_frame)();
    if (r_fext != 0.0) {
        return r_fres / r_fext;
    }
    return r_fres;
}

inline void System::quench()
{
    m_v.fill(0.0);
    m_a.fill(0.0);
    this->updated_v();
}

inline void System::timeStep()
{
    m_inc++;
    xt::noalias(m_v_n) = m_v;
    xt::noalias(m_a_n) = m_a;

    xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
    this->updated_x();

    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
    this->updated_v();

    xt::noalias(m_a) = m_f * m_inv_m;

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    xt::noalias(m_a) = m_f * m_inv_m;

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    xt::noalias(m_a) = m_f * m_inv_m;

    if (xt::any(xt::isnan(m_x))) {
        throw std::runtime_error("NaN entries found");
    }
}

inline void System::timeSteps(size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        this->timeStep();
    }
}

inline size_t System::timeStepsUntilEvent(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList residuals(niter_tol);
    double tol2 = tol * tol;

    auto i_n = this->i();

    for (size_t iiter = 1; iiter < max_iter; ++iiter) {

        this->timeStep();

        for (size_t p = 0; p < m_N; ++p) {
            if (m_y[p].i() != i_n(p)) {
                return iiter;
            }
        }

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return 0;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline void System::flowSteps(size_t n, double v_frame)
{
    for (size_t i = 0; i < n; ++i) {
        m_x_frame += v_frame * m_dt;
        this->timeStep();
    }
}

inline size_t System::flowSteps_boundcheck(size_t n, double v_frame, size_t nmargin)
{
    if (!this->all_inbounds_right(nmargin)) {
        return 0;
    }

    for (size_t i = 0; i < n; ++i) {
        m_x_frame += v_frame * m_dt;
        this->timeStep();

        if (!this->all_inbounds_right(nmargin)) {
            return 0;
        }
    }

    return n;
}

inline long System::minimise(size_t nmargin, double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(max_iter < std::numeric_limits<long>::max());

    if (nmargin == 0) {
        return _minimise_nocheck(tol, niter_tol, static_cast<long>(max_iter));
    }

    return _minimise_check(nmargin, tol, niter_tol, static_cast<long>(max_iter));
}

inline long
System::minimise_timeactivity(size_t nmargin, double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(max_iter < std::numeric_limits<long>::max());

    if (nmargin == 0) {
        return _minimise_timeactivity_nocheck(tol, niter_tol, static_cast<long>(max_iter));
    }

    return _minimise_timeactivity_check(nmargin, tol, niter_tol, static_cast<long>(max_iter));
}

inline long
System::minimise_nopassing(size_t nmargin, double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(max_iter < std::numeric_limits<long>::max());

    if (nmargin == 0) {
        return _minimise_nopassing_nocheck(tol, niter_tol, static_cast<long>(max_iter));
    }

    return _minimise_nopassing_check(nmargin, tol, niter_tol, static_cast<long>(max_iter));
}

inline long System::_minimise_nocheck(double tol, size_t niter_tol, long max_iter)
{
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline long System::_minimise_check(size_t nmargin, double tol, size_t niter_tol, long max_iter)
{
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if (!this->all_inbounds(nmargin)) {
            return -iiter;
        }

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline long System::_minimise_timeactivity_nocheck(double tol, size_t niter_tol, long max_iter)
{
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    auto i_n = this->i();
    long s = 0;
    long s_n = 0;
    bool init = true;

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();

        s = 0;

        for (size_t p = 0; p < m_N; ++p) {
            s += std::abs(m_y[p].i() - i_n(p));
        }

        if (s != s_n) {
            if (init) {
                init = false;
                m_qs_inc_first = m_inc;
            }
            m_qs_inc_last = m_inc;
        }

        s_n = s;

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline long
System::_minimise_timeactivity_check(size_t nmargin, double tol, size_t niter_tol, long max_iter)
{
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    auto i_n = this->i();
    long s = 0;
    long s_n = 0;
    long first_iter = 0;
    long last_iter = 0;
    bool init = true;

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();

        s = 0;

        for (size_t p = 0; p < m_N; ++p) {
            s += std::abs(m_y[p].i() - i_n(p));
        }

        if (s != s_n) {
            if (init) {
                init = false;
                m_qs_inc_first = m_inc;
            }
            m_qs_inc_last = m_inc;
        }

        s_n = s;

        residuals.roll_insert(this->residual());

        if (!this->all_inbounds(nmargin)) {
            return -iiter;
        }

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline size_t System::quasistaticActivityFirst() const
{
    return m_qs_inc_first;
}

inline size_t System::quasistaticActivityLast() const
{
    return m_qs_inc_last;
}

inline long System::_minimise_nopassing_nocheck(double tol, size_t niter_tol, long max_iter)
{
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    double xneigh;
    double x;
    double xmin;
    long i;

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        // "misuse" unused variable
        xt::noalias(m_v_n) = m_x;

        for (size_t p = 0; p < m_N; ++p) {
            // first assuming the particle is always in its local minimum
            if (p == 0) {
                xneigh = m_v_n.back() + m_v_n(1);
            }
            else if (p == m_N - 1) {
                xneigh = m_v_n(m_N - 2) + m_v_n.front();
            }
            else {
                xneigh = m_v_n(p - 1) + m_v_n(p + 1);
            }
            i = m_y[p].i();
            while (true) {
                xmin = 0.5 * (m_y[p].yright() + m_y[p].yleft());
                x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame + m_mu * xmin) /
                    (2 * m_k_neighbours + m_k_frame + m_mu);
                m_y[p].set_x(x);
                if (m_y[p].i() == i) {
                    break;
                }
                i = m_y[p].i();
            }
            m_x(p) = x;
        }

        this->updated_x();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench(); // no dynamics are run: make sure that the user is not confused
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline long
System::_minimise_nopassing_check(size_t nmargin, double tol, size_t niter_tol, long max_iter)
{
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    double xneigh;
    double x;
    double xmin;
    long i;

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        // "misuse" unused variable
        xt::noalias(m_v_n) = m_x;

        for (size_t p = 0; p < m_N; ++p) {
            // first assuming the particle is always in its local minimum
            if (p == 0) {
                xneigh = m_v_n.back() + m_v_n(1);
            }
            else if (p == m_N - 1) {
                xneigh = m_v_n(m_N - 2) + m_v_n.front();
            }
            else {
                xneigh = m_v_n(p - 1) + m_v_n(p + 1);
            }
            i = m_y[p].i();
            while (true) {
                xmin = 0.5 * (m_y[p].yright() + m_y[p].yleft());
                x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame + m_mu * xmin) /
                    (2 * m_k_neighbours + m_k_frame + m_mu);
                m_y[p].set_x(x);
                if (!m_y[p].inbounds(nmargin)) {
                    xt::noalias(m_x) = m_v_n;
                    return -iiter;
                }
                if (m_y[p].i() == i) {
                    break;
                }
                i = m_y[p].i();
            }
            m_x(p) = x;
        }

        this->updated_x();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench(); // no dynamics are run: make sure that the user is not confused
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline double System::advanceUniformly(double dx, bool input_is_frame)
{
    double dx_particles;
    double dx_frame;

    if (input_is_frame) {
        dx_frame = dx;
        dx_particles = dx * m_k_frame / (m_k_frame + m_mu);
    }
    else {
        dx_particles = dx;
        dx_frame = dx * (m_k_frame + m_mu) / m_k_frame;
    }

    m_x += dx_particles;
    m_x_frame += dx_frame;
    this->updated_x();

    if (input_is_frame) {
        return dx_particles;
    }

    return dx_frame;
}

inline double System::eventDrivenStep(double eps, bool kick, int direction)
{
    if (direction > 0 && !kick) {
        double dx = xt::amin(this->yieldDistanceRight())();
        if (dx < 0.5 * eps) {
            return 0.0;
        }
        return this->advanceUniformly(dx - 0.5 * eps, false);
    }

    if (direction > 0 && kick) {
        return this->advanceUniformly(eps, false);
    }

    // direction < 0

    if (!kick) {
        double dx = xt::amin(this->yieldDistanceLeft())();
        if (dx < 0.5 * eps) {
            return 0.0;
        }
        return this->advanceUniformly(0.5 * eps - dx, false);
    }

    return this->advanceUniformly(-eps, false);
}

inline void System::trigger(size_t p, double eps, int direction)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
    if (direction > 0) {
        m_x(p) = m_y[p].yright() + 0.5 * eps;
    }
    else {
        m_x(p) = m_y[p].yleft() - 0.5 * eps;
    }
    this->updated_x();
}

inline size_t System::triggerWeakest(double eps, int direction)
{
    size_t p;

    if (direction > 0) {
        auto v = this->yieldDistanceRight();
        p = std::distance(v.begin(), std::min_element(v.begin(), v.end()));
    }
    else {
        auto v = this->yieldDistanceLeft();
        p = std::distance(v.begin(), std::min_element(v.begin(), v.end()));
    }

    this->trigger(p, eps, direction);
    return p;
}

inline void System::advanceToFixedForce(double f_frame)
{
    auto i_n = this->i();
    this->advanceUniformly((f_frame - xt::mean(m_f_frame)()) / m_mu, false);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::equal(this->i(), i_n)));
}

template <class T, class I>
inline SystemThermalRandomForcing::SystemThermalRandomForcing(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y,
    const I& istart)
{
    this->initSystemThermalRandomForcing(m, eta, mu, k_neighbours, k_frame, dt, x_y, istart);
}

template <class T, class I>
inline void SystemThermalRandomForcing::initSystemThermalRandomForcing(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y,
    const I& istart)
{
    m_seq = false;
    m_f_thermal = xt::zeros<double>({x_y.shape(0)});
    this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, x_y, istart);
}

inline void SystemThermalRandomForcing::updateThermalForce()
{
    if (m_seq) {
        for (size_t p = 0; p < m_N; ++p) {
            if (m_inc >= m_seq_s(p, m_seq_i(p) + 1)) {
                m_seq_i(p)++;
                FRICTIONQPOTSPRINGBLOCK_ASSERT(m_seq_i(p) < m_seq_f.shape(1));
                FRICTIONQPOTSPRINGBLOCK_ASSERT(m_inc < m_seq_s(p, m_seq_i(p) + 1));
            }
            if (m_inc >= m_seq_s(p, m_seq_i(p))) {
                m_f_thermal(p) = m_seq_f(p, m_seq_i(p));
            }
        }
    }
}

inline void SystemThermalRandomForcing::computeForce()
{
    this->updateThermalForce();
    xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame + m_f_thermal;
}

template <class T>
inline void SystemThermalRandomForcing::setRandomForce(const T& f)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(f.dimension() == 1);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(f.size() == m_N);

    m_seq = false;
    m_f_thermal = f;
}

template <class T, class U>
inline void SystemThermalRandomForcing::setRandomForceSequence(const T& f, const U& s)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(f, {m_N, s.shape(1) - 1}));
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(s, {m_N, f.shape(1) + 1}));
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::view(s, xt::all(), 0) <= m_inc));
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::view(s, xt::all(), s.shape(1) - 1) > m_inc));
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::equal(s, xt::sort(s, 1))));

    m_seq = true;
    m_seq_f = f;
    m_seq_s = s;
    m_seq_i = xt::argmax(m_seq_s >= m_inc, 1);

    for (size_t p = 0; p < m_N; ++p) {
        if (m_inc >= m_seq_s(p, m_seq_i(p))) {
            m_f_thermal(p) = m_seq_f(p, m_seq_i(p));
        }
    }
}

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
