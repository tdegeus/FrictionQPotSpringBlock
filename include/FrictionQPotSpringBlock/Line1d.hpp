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

#ifdef XTENSOR_PYTHON_VERSION_MAJOR
    ret.push_back(
        "xtensor-python=" + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MAJOR))) +
        "." + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_PATCH))));
#endif

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

    return ret;
}

template <class T>
inline System::System(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y)
{
    xt::xtensor<long, 1> istart = xt::zeros<long>({x_y.shape(0)});
    this->init(m, eta, mu, k_neighbours, k_frame, dt, x_y, istart);
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
    this->init(m, eta, mu, k_neighbours, k_frame, dt, x_y, istart);
}

template <class T, class I>
inline void System::init(
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
    m_x_t = xt::zeros<double>({m_N});
    m_v_t = xt::zeros<double>({m_N});
    m_a_t = xt::zeros<double>({m_N});
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

inline xt::xtensor<double, 2> System::y()
{
    size_t n = static_cast<size_t>(m_y[0].cend() - m_y[0].cbegin());
    xt::xtensor<double, 2> ret = xt::empty<double>({m_N, n});

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

inline xt::xtensor<double, 1> System::ymin() const
{
    xt::xtensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].ymin();
    }

    return ret;
}

inline xt::xtensor<double, 1> System::ymin_chunk() const
{
    xt::xtensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].ymin_chunk();
    }

    return ret;
}

inline xt::xtensor<double, 1> System::yleft() const
{
    xt::xtensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].yleft();
    }

    return ret;
}

inline xt::xtensor<double, 1> System::yright() const
{
    xt::xtensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].yright();
    }

    return ret;
}

inline xt::xtensor<size_t, 1> System::i_chunk() const
{
    xt::xtensor<size_t, 1> ret = xt::empty<size_t>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].i_chunk();
    }

    return ret;
}

inline xt::xtensor<long, 1> System::istart() const
{
    xt::xtensor<long, 1> ret = xt::empty<long>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].istart();
    }

    return ret;
}

inline xt::xtensor<long, 1> System::istop() const
{
    xt::xtensor<long, 1> ret = xt::empty<long>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].istop();
    }

    return ret;
}

inline xt::xtensor<bool, 1> System::inbounds_left(size_t n) const
{
    xt::xtensor<bool, 1> ret = xt::empty<bool>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].inbounds_left(n);
    }

    return ret;
}

inline xt::xtensor<bool, 1> System::inbounds_right(size_t n) const
{
    xt::xtensor<bool, 1> ret = xt::empty<bool>({m_N});

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

inline xt::xtensor<double, 1> System::yieldDistanceRight() const
{
    return this->yright() - m_x;
}

inline xt::xtensor<double, 1> System::yieldDistanceLeft() const
{
    return m_x - this->yleft();
}

inline xt::xtensor<long, 1> System::i() const
{
    xt::xtensor<long, 1> ret = xt::empty<long>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].i();
    }

    return ret;
}

inline void System::set_t(double arg)
{
    m_t = arg;
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
    return m_t;
}

inline double System::x_frame() const
{
    return m_x_frame;
}

inline xt::xtensor<double, 1> System::x() const
{
    return m_x;
}

inline xt::xtensor<double, 1> System::v() const
{
    return m_v;
}

inline xt::xtensor<double, 1> System::a() const
{
    return m_a;
}

inline xt::xtensor<double, 1> System::f() const
{
    return m_f;
}

inline xt::xtensor<double, 1> System::f_potential() const
{
    return m_f_potential;
}

inline xt::xtensor<double, 1> System::f_frame() const
{
    return m_f_frame;
}

inline xt::xtensor<double, 1> System::f_neighbours() const
{
    return m_f_neighbours;
}

inline xt::xtensor<double, 1> System::f_damping() const
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
    m_t += m_dt;
    xt::noalias(m_v_n) = m_v;
    xt::noalias(m_a_n) = m_a;

    xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
    this->updated_x();

    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
    this->updated_v();

    xt::noalias(m_a) = m_f / m_m;

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    xt::noalias(m_a) = m_f / m_m;

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    xt::noalias(m_a) = m_f / m_m;

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

inline size_t System::minimise(double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline size_t
System::minimise_boundcheck(size_t nmargin, double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);
    xt::noalias(m_x_t) = m_x;
    xt::noalias(m_v_t) = m_v;
    xt::noalias(m_a_t) = m_a;
    m_t_t = m_t;

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }

        if (!this->all_inbounds_right(nmargin)) {
            xt::noalias(m_x) = m_x_t;
            xt::noalias(m_v) = m_v_t;
            xt::noalias(m_a) = m_a_t;
            m_t = m_t_t;
            return 0;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline size_t System::minimise_timeactivity(double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    auto i_n = this->i();
    long s = 0;
    long s_n = 0;
    size_t first_iter = 0;
    size_t last_iter = 0;
    bool init = true;

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();

        s = 0;

        for (size_t p = 0; p < m_N; ++p) {
            s += std::abs(m_y[p].i() - i_n(p));
        }

        if (s != s_n) {
            if (init) {
                init = false;
                first_iter = iiter;
            }
            last_iter = iiter;
        }

        s_n = s;

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return last_iter - first_iter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline size_t System::minimise_timeactivity_boundcheck(
    size_t nmargin,
    double tol,
    size_t niter_tol,
    size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);
    xt::noalias(m_x_t) = m_x;
    xt::noalias(m_v_t) = m_v;
    xt::noalias(m_a_t) = m_a;
    m_t_t = m_t;

    auto i_n = this->i();
    long s = 0;
    long s_n = 0;
    size_t first_iter = 0;
    size_t last_iter = 0;
    bool init = true;

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();

        s = 0;

        for (size_t p = 0; p < m_N; ++p) {
            s += std::abs(m_y[p].i() - i_n(p));
        }

        if (s != s_n) {
            if (init) {
                init = false;
                first_iter = iiter;
            }
            last_iter = iiter;
        }

        s_n = s;

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return last_iter - first_iter;
        }

        if (!this->all_inbounds_right(nmargin)) {
            xt::noalias(m_x) = m_x_t;
            xt::noalias(m_v) = m_v_t;
            xt::noalias(m_a) = m_a_t;
            m_t = m_t_t;
            return 0;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline size_t System::minimise_nopassing(double tol, size_t niter_tol, size_t max_iter)
{
    FRICTIONQPOTSPRINGBLOCK_REQUIRE(tol < 1.0);
    double tol2 = tol * tol;
    GooseFEM::Iterate::StopList residuals(niter_tol);

    double xneigh;
    double x;
    double xmin;

    for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

        for (size_t p = 0; p < m_N; ++p) {
            // first assuming the particle is always in its local minimum
            if (p == 0) {
                xneigh = m_x.back() + m_x(1);
            }
            else if (p == m_N - 1) {
                xneigh = m_x(m_N - 2) + m_x.front();
            }
            else {
                xneigh = m_x(p - 1) + m_x(p + 1);
            }
            x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame) /
                (2 * m_k_neighbours + m_k_frame);
            m_y[p].set_x(x);
            // then fine tuning based on local potential
            xmin = 0.5 * (m_y[p].yright() + m_y[p].yleft());
            x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame + m_mu * xmin) /
                (2 * m_k_neighbours + m_k_frame + m_mu);
            m_y[p].set_x(x);
            m_x(p) = x;
        }

        this->updated_x();
        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
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
        p = xt::argmin(this->yieldDistanceRight())();
    }
    else {
        p = xt::argmin(this->yieldDistanceLeft())();
    }

    this->trigger(p, eps, direction);
    return p;
}

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
