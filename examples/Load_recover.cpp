
#include <FrictionQPotSpringBlock/Line1d.h>
#include <QPot/random.hpp>
#include <fmt/core.h>
#include <GooseFEM/Iterate.h>
#include <xtensor/xrandom.hpp>
#include <highfive/H5Easy.hpp>

#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)

void load()
{
    H5Easy::File data("Load_recover.h5", H5Easy::File::Overwrite);

    size_t seed = static_cast<size_t>(time(NULL));
    size_t N = 1000;
    double alpha = 0.5;
    double beta = 2.0 / alpha;
    auto gamma = QPot::random::GammaList(alpha, beta);
    QPot::random::seed(seed);

    double xdelta = 1e-3;

    H5Easy::dump(data, "/meta/N", N);
    H5Easy::dump(data, "/meta/alpha", alpha);
    H5Easy::dump(data, "/meta/beta", beta);
    H5Easy::dump(data, "/meta/seed", seed);

    GooseFEM::Iterate::StopList stop(10);

    FrictionQPotSpringBlock::Line1d::System sys(N, gamma);

    sys.set_dt(0.1);
    sys.set_eta(2.0 * std::sqrt(3.0) / 10.0);
    sys.set_m(1.0);
    sys.set_mu(1.0);
    sys.set_k_neighbours(1.0);
    sys.set_k_frame(1.0 / double(N));

    size_t ninc = 1000;
    bool redraw;
    size_t nredraw = 0;

    for (size_t inc = 0; inc < ninc; ++inc) {

        // Apply event-driven protocol.
        if (inc == 0) {
            sys.set_x_frame(0.0); // initial quench
        }
        else if (inc % 2 != 0) {
            redraw = sys.advanceRightElastic(xdelta); // elastically advance -> mechanical equilibrium
        }
        else {
            redraw = sys.advanceRightKick(xdelta); // apply kick
        }

        // Save recovery information.
        if (redraw) {
            H5Easy::dump(data, "/redraw/stored", nredraw, {nredraw});
            H5Easy::dump(data, "/redraw/inc", inc, {nredraw});
            H5Easy::dump(data, fmt::format("/redraw/{0:d}", nredraw), sys.x());
            nredraw++;
        }

        // Minimise energy.
        if (inc % 2 == 0) {
            size_t niter = 0;

            for (;; ++niter) {

                redraw = sys.timeStep();

                if (redraw) {
                    H5Easy::dump(data, "/redraw/stored", nredraw, {nredraw});
                    H5Easy::dump(data, "/redraw/inc", inc, {nredraw});
                    H5Easy::dump(data, fmt::format("/redraw/{0:d}", nredraw), sys.x());
                    nredraw++;
                }

                if (stop.stop(sys.residual(), 1e-5)) {
                    sys.quench();
                    break;
                }

                if (niter > 100000) {
                    throw std::runtime_error("No convergence found");
                }
            }

            fmt::print("inc = {0:4d}, niter = {1:d}\n", inc, niter);
        }

        // Extract output data.
        H5Easy::dump(data, "/stored", inc, {inc});
        H5Easy::dump(data, "/x_frame", sys.x_frame(), {inc});
        H5Easy::dump(data, "/f_frame", xt::mean(sys.f_frame())(), {inc});
        H5Easy::dump(data, fmt::format("/x/{0:d}", inc), sys.x());
        H5Easy::dump(data, fmt::format("/yieldLeft/{0:d}", inc), sys.yieldLeft());
        H5Easy::dump(data, fmt::format("/yieldRight/{0:d}", inc), sys.yieldRight());
    }
}

void recover()
{
    H5Easy::File data("Load_recover.h5", H5Easy::File::ReadOnly);

    auto N = H5Easy::load<size_t>(data, "/meta/N");
    auto alpha = H5Easy::load<double>(data, "/meta/alpha");
    auto beta = H5Easy::load<double>(data, "/meta/beta");
    auto gamma = QPot::random::GammaList(alpha, beta);
    QPot::random::seed(H5Easy::load<size_t>(data, "/meta/seed"));

    FrictionQPotSpringBlock::Line1d::System sys(N, gamma);

    auto incs = H5Easy::load<xt::xtensor<size_t, 1>>(data, "/stored");
    auto redraw = H5Easy::load<xt::xtensor<size_t, 1>>(data, "/redraw/inc");
    size_t iredraw = 0;

    for (auto& inc : incs) {

        // set position used at time of redraw, if needed
        while (true) {
            if (iredraw == redraw.size()) {
                break;
            }
            if (redraw(iredraw) > inc) {
                break;
            }
            MYASSERT(sys.set_x(
                H5Easy::load<xt::xtensor<double, 1>>(data, fmt::format("/redraw/{0:d}", iredraw))));
            iredraw++;
        }

        // the system is now fully determined, and deterministic
        MYASSERT(!sys.set_x(
            H5Easy::load<xt::xtensor<double, 1>>(data, fmt::format("/x/{0:d}", inc))));

        MYASSERT(xt::allclose(sys.yieldLeft(),
            H5Easy::load<xt::xtensor<double, 1>>(data, fmt::format("/yieldLeft/{0:d}", inc))));

        MYASSERT(xt::allclose(sys.yieldRight(),
            H5Easy::load<xt::xtensor<double, 1>>(data, fmt::format("/yieldRight/{0:d}", inc))));
    }

    std::cout << "Successful recover" << std::endl;
}

int main()
{
    load();
    recover();
}
