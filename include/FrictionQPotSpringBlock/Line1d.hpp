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

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

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

inline void System::initiate_gamma(double arg)
{
    m_gamma = arg;
}

inline void System::initiate_lns(double arg)
{
    m_lns = arg;
}

inline void System::initiate_temperature()
{
    computeTemperature(m_v);
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

inline void System::set_kBoltzmann(double arg)
{
    m_kBoltzmann = arg;
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

inline void System::set_Q(double arg)
{
    m_Q = arg;
}

inline void System::set_temperature(double arg)
{
    m_temperature_set = arg;
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

void System::computeTemperature(xt::xtensor<double, 1> velocities)
{
    m_temperature_inst = xt::norm_sq(velocities)();
    m_temperature_inst = m_temperature_inst * m_m / ( m_kBoltzmann * m_N );
}

inline double System::compute_NHenergy()
{
    double inst_nh_energy = 0.0;
    inst_nh_energy = m_en_kinetic + m_en_potential //
        + 0.5 * std::pow(m_gamma,2) * m_Q //
        + m_N * m_kBoltzmann * m_temperature_set * m_lns;
    return inst_nh_energy;
}

inline double System::compute_kinetic_energy()
{
    double te_Ek;
    te_Ek = xt::norm_sq(m_v)() * m_m * 0.5;
    return te_Ek;
}

inline double System::compute_potential_energy()
{
    double te_Ep = 0.0;
    for (size_t p = 0; p < m_N; ++p) {
        double x = m_x(p);
        m_y[p].set_x(x);
        te_Ep += 0.5 * m_mu * std::pow(x - 0.5 * (m_y[p].yright() + m_y[p].yleft()),2);
    }
    return te_Ep;
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

    m_ave_force_frame = step_force_frame();
    //m_x_frame += m_frame_speed;

    if (xt::any(xt::isnan(m_x))) {
        throw std::runtime_error("NaN entries found");
    }
}

inline void System::NHtimestep(double m_frame_speed, size_t m_nchunk)
{
    
    if (m_temperature_set > 0)
    { 
        m_t += m_dt;
        xt::noalias(m_v_n) = m_v;

        //positions
        xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * (m_a - m_gamma * m_v);
        //velocities
        xt::noalias(m_v) = m_v + m_dt * 0.5 * (m_a - m_gamma * m_v);
    
        //compute gamma and lnS
        this->computeTemperature(m_v_n);
        m_gamma += m_dt * 0.5 * m_kBoltzmann * (m_N * m_temperature_inst - (m_N + 1) * m_temperature_set) / m_Q;
        //m_lns += m_gamma * m_dt * 0.5 //
        //    + 0.125 * std::pow(m_dt,2) * m_kBoltzmann * (m_N * m_temperature_inst - (m_N + 1) * m_temperature_set) / m_Q;
        this->computeTemperature(m_v);
        m_gamma += m_dt * 0.5 * m_kBoltzmann * (m_N * m_temperature_inst - (m_N + 1) * m_temperature_set) / m_Q;
        //m_lns += m_gamma * m_dt * 0.5 //
        //    + 0.125 * std::pow(m_dt,2) * m_kBoltzmann * (m_N * m_temperature_inst - (m_N + 1) * m_temperature_set) / m_Q;

        //compute force and accelerations
        this->computeForcePotential();
        this->computeForceNeighbours();
        this->computeForceFrame();
        this->computeForceDamping();
        this->computeForce();
        xt::noalias(m_a) = m_f / m_m;

        //velocities
        xt::noalias(m_v_n) = m_v;
        xt::noalias(m_v) = (m_v_n + m_dt * 0.5 * m_a) / (1 + m_dt * 0.5 * m_gamma);

        // accelerations      
        m_ave_force_frame = step_force_frame();
        m_x_frame += m_frame_speed;
        this->computeForceDamping();
        this->computeForceFrame();
        this->computeForce();
        xt::noalias(m_a) = m_f / m_m;

        if (xt::any(xt::isnan(m_x))) {
            throw std::runtime_error("NaN entries found");
        }

        if (xt::any(this->i() > m_nchunk - 10)) {
            throw std::runtime_error("explode!");
        }
    }

    if (m_temperature_set == 0)
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
        this->computeTemperature(m_v);

        // accelerations
        this->computeForce();
        xt::noalias(m_a) = m_f / m_m;

        m_ave_force_frame = step_force_frame();
        m_x_frame += m_frame_speed;

        if (xt::any(xt::isnan(m_x))) {
            throw std::runtime_error("NaN entries found");
        }

        if (xt::any(this->i() > m_nchunk - 10)) {
            throw std::runtime_error("explode!");
        }
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

inline size_t System::minimise(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList residuals(niter_tol);
    double tol2 = tol * tol;

    for (size_t iiter = 0; iiter < max_iter; ++iiter) {

        this->timeStep();

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}


inline size_t System::minimise_timeactivity(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList residuals(niter_tol);
    double tol2 = tol * tol;

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

        residuals.roll_insert(this->residual());

        if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
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
    FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON(
        "rename 'advanceRightElastic' -> 'advanceEventRightElastic'");
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
        FRICTIONQPOTSPRINGBLOCK_DEBUG(
            std::abs(dx_particles) < xt::amin(this->yieldDistanceLeft())());
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

inline void System::printStep(size_t runstep)
{
    printf("step:%u\ttemperature:%3f\tframe force:%3f\n",runstep,m_temperature_inst,m_ave_force_frame);
}

inline double System::output_temperature()
{
    return m_temperature_inst;
}

inline double System::output_force_frame()
{
    return m_ave_force_frame;
}

inline double System::output_force_potential()
{
    double temp_force = 0.0;
    for (size_t p = 0; p < m_N; ++p) {
        temp_force += m_f_potential(p);
    }
    m_ave_force_potential = temp_force;
    return temp_force;
}

inline double System::output_position()
{
    m_ave_position = xt::mean(this->i())();
    return m_ave_position;
}

inline double System::step_force_frame()
{
    double temp_force = 0.0;
    for (size_t p = 0; p < m_N; ++p) {
        temp_force += m_f_frame(p);
    }
    m_ave_force_frame = temp_force;
    return temp_force;
}

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
