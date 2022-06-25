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
    this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, x_y);
}

template <class T>
inline void System::initSystem(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 2);

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
    m_i = xt::zeros<long>({m_N}); // consistent with `lower_bound`
    m_y = x_y;

    this->updated_x();
    this->updated_v();
}

inline size_t System::N() const
{
    return m_N;
}

inline const array_type::tensor<double, 2>& System::y()
{
    return m_y;
}

inline const array_type::tensor<long, 1>& System::i() const
{
    return m_i;
}

inline array_type::tensor<double, 1> System::yieldDistanceRight() const
{
    array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y(p, m_i(p) + 1) - m_x(p);
    }

    return ret;
}

inline array_type::tensor<double, 1> System::yieldDistanceLeft() const
{
    array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_x(p) - m_y(p, m_i(p));
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
inline void System::set_y(const T& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(arg.dimension() == 2);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(arg.shape(0) == m_N);
    xt::noalias(m_y) = arg;
    this->updated_x();
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

inline const array_type::tensor<double, 1>& System::x() const
{
    return m_x;
}

inline const array_type::tensor<double, 1>& System::v() const
{
    return m_v;
}

inline const array_type::tensor<double, 1>& System::a() const
{
    return m_a;
}

inline const array_type::tensor<double, 1>& System::f() const
{
    return m_f;
}

inline const array_type::tensor<double, 1>& System::f_potential() const
{
    return m_f_potential;
}

inline const array_type::tensor<double, 1>& System::f_frame() const
{
    return m_f_frame;
}

inline const array_type::tensor<double, 1>& System::f_neighbours() const
{
    return m_f_neighbours;
}

inline const array_type::tensor<double, 1>& System::f_damping() const
{
    return m_f_damping;
}

inline void System::refresh()
{
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForceDamping();
    this->computeForce();
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
    QPot::inplace::lower_bound(m_y, m_x, m_i);

    for (size_t p = 0; p < m_N; ++p) {
        auto* l = &m_y(p, m_i(p));
        m_f_potential(p) = m_mu * (0.5 * (*(l) + *(l + 1)) - m_x(p));
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

    auto i_n = m_i;

    for (size_t iiter = 1; iiter < max_iter; ++iiter) {

        this->timeStep();

        if (xt::any(xt::not_equal(m_i, i_n))) {
            return iiter;
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
    auto nyield = m_y.shape(1);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(nmargin < nyield);

    if (xt::any(m_i > nyield - nmargin)) {
        return 0;
    }

    for (size_t i = 0; i < n; ++i) {
        m_x_frame += v_frame * m_dt;
        this->timeStep();

        if (xt::any(m_i > nyield - nmargin)) {
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
    auto nyield = m_y.shape(1);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(nmargin < nyield);

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();
        residuals.roll_insert(this->residual());

        if (xt::any(m_i < nmargin) || xt::any(m_i > nyield - nmargin)) {
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

    auto i_n = m_i;
    long s = 0;
    long s_n = 0;
    bool init = true;

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();

        s = xt::sum(xt::abs(m_i - i_n))();

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

    auto i_n = m_i;
    long s = 0;
    long s_n = 0;
    bool init = true;
    auto nyield = m_y.shape(1);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(nmargin < nyield);

    for (long iiter = 1; iiter < max_iter + 1; ++iiter) {

        this->timeStep();

        s = xt::sum(xt::abs(m_i - i_n))();

        if (s != s_n) {
            if (init) {
                init = false;
                m_qs_inc_first = m_inc;
            }
            m_qs_inc_last = m_inc;
        }

        s_n = s;

        residuals.roll_insert(this->residual());

        if (xt::any(m_i < nmargin) || xt::any(m_i > nyield - nmargin)) {
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
    auto nyield = m_y.shape(1);

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
            auto i = m_i(p);
            auto* l = &m_y(p, i);
            auto* y = &m_y(p, 0);
            while (true) {
                xmin = 0.5 * (*(l) + *(l + 1));
                x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame + m_mu * xmin) /
                    (2 * m_k_neighbours + m_k_frame + m_mu);
                m_i(p) = QPot::iterator::lower_bound(y, y + nyield, x, i);
                if (m_i(p) == i) {
                    break;
                }
                i = m_i(p);
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
    auto nyield = m_y.shape(1);

    FRICTIONQPOTSPRINGBLOCK_ASSERT(nyield > nmargin);

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
            i = m_i(p);
            auto* l = &m_y(p, i);
            auto* y = &m_y(p, 0);
            while (true) {
                xmin = 0.5 * (*(l) + *(l + 1));
                x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame + m_mu * xmin) /
                    (2 * m_k_neighbours + m_k_frame + m_mu);
                m_i(p) = QPot::iterator::lower_bound(y, y + nyield, x, i);
                if (static_cast<size_t>(m_i(p)) > nyield - nmargin) {
                    xt::noalias(m_x) = m_v_n;
                    return -iiter;
                }
                if (m_i(p) == i) {
                    break;
                }
                i = m_i(p);
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
        m_x(p) = m_y(p, m_i(p) + 1) + 0.5 * eps;
    }
    else {
        m_x(p) = m_y(p, m_i(p)) - 0.5 * eps;
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

template <class T>
inline SystemThermalRandomForcing::SystemThermalRandomForcing(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y)
{
    this->initSystemThermalRandomForcing(m, eta, mu, k_neighbours, k_frame, dt, x_y);
}

template <class T>
inline void SystemThermalRandomForcing::initSystemThermalRandomForcing(
    double m,
    double eta,
    double mu,
    double k_neighbours,
    double k_frame,
    double dt,
    const T& x_y)
{
    m_seq = false;
    m_f_thermal = xt::zeros<double>({x_y.shape(0)});
    this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, x_y);
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
