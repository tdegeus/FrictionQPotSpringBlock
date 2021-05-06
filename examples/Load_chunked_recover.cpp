#include <FrictionQPotSpringBlock/Line1d.h>
#include <prrng.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>

int main()
{
    H5Easy::File data("Load_chunked_recover.h5", H5Easy::File::Overwrite);

    size_t N = 1000;

    xt::xtensor<size_t, 1> initstate = xt::arange<size_t>(N);
    xt::xtensor<size_t, 1> initseq = xt::zeros<size_t>({N});
    auto generators = prrng::auto_pcg32(initstate, initseq);
    auto regenerators = prrng::auto_pcg32(initstate, initseq);
    auto state = generators.state();
    xt::xtensor<long, 1> istate = xt::zeros<long>({N});

    H5Easy::dump(data, "/meta/N", N);
    H5Easy::dump(data, "/meta/initstate", initstate);
    H5Easy::dump(data, "/meta/initseq", initseq);

    size_t nchunk = 2000; // size of chunk of yield positions kept in memory
    size_t nbuffer = 100; // buffer to keep left

    xt::xtensor<double, 2> y = 2.0 * generators.random({nchunk});
    y = xt::cumsum(y, 1);
    y -= 50.0;

    double xdelta = 1e-3;

    FrictionQPotSpringBlock::Line1d::System sys(N, y);

    sys.set_dt(0.1);
    sys.set_eta(2.0 * std::sqrt(3.0) / 10.0);
    sys.set_m(1.0);
    sys.set_mu(1.0);
    sys.set_k_neighbours(1.0);
    sys.set_k_frame(1.0 / double(N));

    size_t ninc = 1000;

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

        // Change the chunk of yield positions held in memory,
        // such that "nbuffer" yield positions are held left of the current "x"
        // (maximise the size of the chunk in positive direction)
        // Reminder: a sequence of yield positions is taken from the random generator:
        //           each particle has one generator, with its own seed, and one sequence
        // Note:     the assumptions are made that:
        //           - no shift is needed during minimisation
        //           - one shift suffices here
        if (xt::any(sys.i_chunk() > nbuffer)) {
            for (size_t p = 0; p < N; ++p) {
                QPot::Chunked& yp = sys.y(p);
                auto nb = yp.size() - yp.i_chunk() + nbuffer;
                if (nb >= nchunk) {
                    continue;
                }
                state(p) = generators[p].state();
                istate(p) = yp.istop();
                yp.shift_dy(yp.istop(), xt::eval(2.0 * generators[p].random({nchunk - nb})), nb);
            }
        }

        // to restore state: start with the latests draw that is quite close and reverse back
        // in the sequence until the start of the current chunk held in memory
        regenerators.restore(state);
        regenerators.advance(xt::eval(sys.istart() - istate));

        // Extract output data.
        H5Easy::dump(data, "/stored", inc, {inc});
        H5Easy::dump(data, "/x_frame", sys.x_frame(), {inc});
        H5Easy::dump(data, fmt::format("/x/{0:d}", inc), sys.x());
        H5Easy::dump(data, fmt::format("/ymin/{0:d}", inc), sys.ymin());
        H5Easy::dump(data, fmt::format("/istart/{0:d}", inc), sys.istart());
        H5Easy::dump(data, fmt::format("/state/{0:d}", inc), regenerators.state());
    }

    return 0;
}
