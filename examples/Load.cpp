
#include <FrictionQPotSpringBlock/Line1d.h>
#include <xtensor/xrandom.hpp>
#include <xtensor/xcsv.hpp>
#include <fstream>

int main()
{
    size_t N = 1000;
    double alpha = 0.5;
    double beta = 2.0 / alpha;
    auto gamma = [=](std::vector<size_t> shape) {
        return xt::random::gamma<double>(shape, alpha, beta); };

    double xdelta = 1e-3;

    FrictionQPotSpringBlock::Line1d::System sys(N, gamma);

    sys.set_dt(0.1);
    sys.set_eta(2.0 * std::sqrt(3.0) / 10.0);
    sys.set_m(1.0);
    sys.set_mu(1.0);
    sys.set_k_neighbours(1.0);
    sys.set_k_frame(1.0 / double(N));

    size_t ninc = 1000;
    xt::xtensor<double, 2> ret = xt::empty<double>({size_t(2), ninc});

    for (size_t inc = 0; inc < ninc; ++inc) {

        // Apply event-driven protocol.
        if (inc == 0) {
            sys.set_x_frame(0.0); // initial quench
        }
        else if (inc % 2 != 0) {
            sys.advanceRightElastic(xdelta); // elastically advance -> mechanical equilibrium
        }
        else {
            sys.advanceRightKick(xdelta); // apply kick
        }

        // Minimise energy.
        if (inc % 2 == 0) {
            size_t niter = sys.minimise();
            fmt::print("inc = {0:4d}, niter = {1:d}\n", inc, niter);
        }

        // Extract output data.
        ret(0, inc) = sys.get_x_frame();
        ret(1, inc) = xt::mean(sys.get_f_frame())();
    }

    std::ofstream outfile("Load.txt");
    xt::dump_csv(outfile, ret);
    outfile.close();

    return 0;
}
