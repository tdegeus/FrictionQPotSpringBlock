/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotSpringBlock

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

    ret.push_back("qpot=" + QPot::version());

    ret.push_back("goosefem=" + GooseFEM::version());

    ret.push_back("xtensor=" +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

    return ret;
}

template <class T>
inline System::System(size_t N, const T& x_y)
{
    xt::xtensor<long, 1> istart = xt::zeros<long>({N});
    this->init(N, x_y, istart);
}

template <class T, class I>
inline System::System(size_t N, const T& x_y, const I& istart)
{
    this->init(N, x_y, istart);
}

template <class T, class I>
inline void System::init(size_t N, const T& x_y, const I& istart)
{
    #ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 2);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(istart.dimension() == 1);
    auto y0 = xt::view(x_y, xt::all(), 0);
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(y0 < 0.0));
    #endif

    m_N = N;
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

    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForceDamping();
    this->computeForce();
}

inline QPot::Chunked& System::y(size_t p)
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
    FRICTIONQPOTSPRINGBLOCK_ASSERT(x_y.dimension() == 1);
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

inline xt::xtensor<bool, 1> System::boundcheck_left(size_t n) const
{
    xt::xtensor<bool, 1> ret = xt::empty<bool>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].boundcheck_left(n);
    }

    return ret;
}

inline xt::xtensor<bool, 1> System::boundcheck_right(size_t n) const
{
    xt::xtensor<bool, 1> ret = xt::empty<bool>({m_N});

    for (size_t p = 0; p < m_N; ++p) {
        ret(p) = m_y[p].boundcheck_right(n);
    }

    return ret;
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

inline bool System::any_shift(size_t n) const
{
    for (size_t p = 0; p < m_N; ++p) {
        if (!m_y[p].boundcheck_left(n) || !m_y[p].boundcheck_right(n)) {
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

inline size_t System::N() const
{
    return m_N;
}

inline void System::set_t(double arg)
{
    m_t = arg;
}

inline void System::set_dt(double arg)
{
    m_dt = arg;
}

inline void System::set_eta(double arg)
{
    m_eta = arg;
}

inline void System::set_m(double arg)
{
    m_m = arg;
}

inline void System::set_mu(double arg)
{
    m_mu = arg;
}

inline void System::set_k_neighbours(double arg)
{
    m_k_neighbours = arg;
}

inline void System::set_k_frame(double arg)
{
    m_k_frame = arg;
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
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForce();
}

template <class T>
inline void System::set_v(const T& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
    xt::noalias(m_v) = arg;
    this->computeForceDamping();
    this->computeForce();
}

template <class T>
inline void System::set_a(const T& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
    xt::noalias(m_a) = arg;
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
    m_f_neighbours(0) = m_k_neighbours * (m_x(m_N - 1) - 2 * m_x(0) + m_x(1));
    m_f_neighbours(m_N - 1) = m_k_neighbours * (m_x(m_N - 2) - 2 * m_x(m_N - 1) + m_x(0));
}

inline void System::computeForceFrame()
{
    xt::noalias(m_f_frame) = m_k_frame * (m_x_frame - m_x);
}

inline void System::computeForceDamping()
{
    xt::noalias(m_f_damping) = - m_eta * m_v;
}

inline void System::computeForce()
{
    xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame;
}

inline void System::timeStep()
{
    m_t += m_dt;
    xt::noalias(m_v_n) = m_v;
    xt::noalias(m_a_n) = m_a;

    // positions
    xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();

    // velocities
    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
    this->computeForceDamping();

    // accelerations
    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    // velocities
    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->computeForceDamping();

    // accelerations
    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    // velocities
    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->computeForceDamping();

    // accelerations
    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    if (xt::any(xt::isnan(m_x))) {
        throw std::runtime_error("NaN entries found");
    }
}

inline double System::t() const
{
    return m_t;
}

inline double System::dt() const
{
    return m_dt;
}

inline double System::residual() const
{
    double Fext = xt::norm_l2(m_f_frame)();
    double Fnorm = (Fext < std::numeric_limits<double>::epsilon()) ? 1 : Fext;
    return xt::norm_l2(m_f)() / Fnorm;
}

inline void System::quench()
{
    m_v.fill(0.0);
    m_a.fill(0.0);
    this->computeForceDamping();
    this->computeForce();
}

inline size_t System::timeStepsUntilEvent(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList stop(niter_tol);

    auto i_n = this->i();

    for (size_t iiter = 1; iiter < max_iter; ++iiter) {

        this->timeStep();

        for (size_t p = 0; p < m_N; ++p) {
            if (m_y[p].i() != i_n(p)) {
                return iiter;
            }
        }

        if (stop.stop_simple(this->residual(), tol)) {
            this->quench();
            return 0;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline size_t System::minimise(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList stop(niter_tol);

    for (size_t iiter = 0; iiter < max_iter; ++iiter) {

        this->timeStep();

        if (stop.stop(this->residual(), tol)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline size_t System::minimise_timeactivity(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList stop(niter_tol);

    auto i_n = this->i();
    long s = 0;
    long s_n = 0;
    size_t first_iter = 0;
    size_t last_iter = 0;
    bool init = true;

    for (size_t iiter = 0; iiter < max_iter; ++iiter) {

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

        if (stop.stop(this->residual(), tol)) {
            this->quench();
            return last_iter - first_iter;
        }
    }

    throw std::runtime_error("No convergence found");
}

/**
\cond
*/

inline void System::advanceRightElastic(double eps)
{
    FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON("rename 'advanceRightElastic' -> 'advanceEventRightElastic'");
    this->advanceEventRightElastic(eps);
}

inline void System::advanceRightKick(double eps)
{
    FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON("rename 'advanceRightKick' -> 'advanceEventRightKick'");
    this->advanceEventRightKick(eps);
}

/**
\endcond
*/

inline void System::advanceEventRightElastic(double eps)
{
    double dx = xt::amin(this->yieldDistanceRight())();
    if (dx < eps / 2.0) {
        return;
    }
    this->advanceElastic(dx - eps / 2.0, false);
}

inline void System::advanceEventRightKick(double eps)
{
    this->advanceElastic(eps, false);
}

inline void System::advanceElastic(double dx, bool dx_of_frame)
{
    double dx_particles;
    double dx_frame;

    if (dx_of_frame) {
        dx_frame = dx;
        dx_particles = dx * m_k_frame / (m_k_frame + m_mu);
    }
    else {
        dx_particles = dx;
        dx_frame = dx * (m_k_frame + m_mu) / m_k_frame;
    }

    #ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_DEBUG
        if (dx_particles > 0.0) {
            FRICTIONQPOTSPRINGBLOCK_DEBUG(dx_particles < xt::amin(this->yieldDistanceRight())());
        }
        else {
            FRICTIONQPOTSPRINGBLOCK_DEBUG(std::abs(dx_particles) < xt::amin(this->yieldDistanceLeft())());
        }
    #endif

    m_x += dx_particles;
    m_x_frame += dx_frame;
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForce();
}

inline void System::triggerRight(size_t p, double eps)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
    m_x(p) = m_y[p].yright() + eps / 2.0;
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForce();
}

inline void System::triggerWeakestRight(double eps)
{
    this->triggerRight(xt::argmin(this->yieldDistanceRight())(), eps);
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

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
