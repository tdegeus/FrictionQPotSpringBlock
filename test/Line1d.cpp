
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotSpringBlock/Line1d.h>
#include <prrng.h>

TEST_CASE("FrictionQPotSpringBlock::Line1d", "Line1d.h")
{

SECTION("System::advanceRightElastic")
{
    size_t N = 3;
    xt::xtensor<double, 2> y = xt::ones<double>({N, size_t(100)});
    y = xt::cumsum(y, 1);
    y -= 49.5;

    FrictionQPotSpringBlock::Line1d::System sys(N, y);

    sys.advanceRightElastic(0.2);

    xt::xtensor<double, 1> x = (0.5 - 0.1) * xt::ones<double>({N});

    REQUIRE(xt::allclose(x, sys.x()));
}

SECTION("Reconstruct sequence")
{
    size_t N = 3;
    auto seed = time(NULL);
    xt::xtensor<size_t, 1> initstate = seed + xt::arange<size_t>(N);

    size_t nchunk = 100; // size of chunk of yield positions kept in memory
    size_t nbuffer = 40; // buffer when shifting chunks of yield positions
    size_t ncheck = 15; // boundary region to check of chunk-shifting is needed
    size_t nmax = 20; // maximal boundary region for which chunk-shifting is applied
    double init_offset = 50.0; // initial negative position shift

    // allocate generators
    auto generators = prrng::auto_pcg32(initstate);
    auto regenerators = prrng::auto_pcg32(initstate);

    // store the current state of the generators
    auto state = generators.state();

    // draw initial chunk from the generators and convert to yield positions
    xt::xtensor<double, 2> y = 2.0 * generators.random({nchunk});
    y = xt::cumsum(y, 1);
    y -= init_offset;

    // state information of the first yield position of each particle (for restoring later)
    xt::xtensor<long, 1> istart = xt::zeros<long>({N});
    xt::xtensor<double, 1> ymin = xt::view(y, xt::all(), 0);

    // keep history to the state to facilitate restoring the right chunk for a given position
    // (the buffer can make that the current position is in the last chunk)
    auto state_n = state;
    auto istart_n = istart;
    auto ymin_n = ymin;

    // initialise system
    FrictionQPotSpringBlock::Line1d::System sys(N, y);
    FrictionQPotSpringBlock::Line1d::System resys(N, y);

    size_t n = 50;
    xt::xtensor<double, 1> x = 10.0 * xt::ones<double>({N});
    x(0) = 5.0;
    x(1) = 7.0;

    for (size_t i = 0; i < n; ++i) {

        sys.set_x(xt::eval((double)(i) * x));

        if (sys.any_shift(ncheck)) {
            auto l = sys.boundcheck_left(nmax);
            auto r = sys.boundcheck_right(nmax);

            REQUIRE(xt::all(l));

            for (size_t p = 0; p < N; ++p) {
                if (!r(p)) {
                    QPot::Chunked& yp = sys.y(p);

                    state_n(p) = state(p);
                    istart_n(p) = istart(p);
                    ymin_n(p) = ymin(p);

                    state(p) = generators[p].state();
                    istart(p) += nchunk;

                    yp.shift_dy(
                        istart(p),
                        xt::eval(2.0 * generators[p].random({nchunk})),
                        nbuffer);

                    ymin(p) = yp.ymin_chunk();
                }
            }

            REQUIRE(!sys.any_shift(ncheck));
        }

        regenerators.restore(state_n);
        xt::xtensor<double, 2> ry = 2.0 * regenerators.random({2 * nchunk});
        xt::view(ry, xt::all(), 0) = ymin_n;
        ry = xt::cumsum(ry, 1);
        resys.set_y(istart_n, ry);
        resys.set_x(sys.x());

        REQUIRE(xt::allclose(sys.yieldIndex(), resys.yieldIndex()));
        REQUIRE(xt::allclose(sys.yieldDistanceLeft(), resys.yieldDistanceLeft()));
        REQUIRE(xt::allclose(sys.yieldDistanceRight(), resys.yieldDistanceRight()));
        REQUIRE(xt::allclose(sys.yleft(), resys.yleft()));
        REQUIRE(xt::allclose(sys.yright(), resys.yright()));
    }
}

// SECTION("Reconstruct sequence minimal data")
// {
//     size_t N = 3;
//     auto seed = time(NULL);
//     auto random = QPot::random::RandList();

//     QPot::random::seed(seed);
//     FrictionQPotSpringBlock::Line1d::System sys(N, random);

//     size_t n = 20;
//     xt::xtensor<double, 1> x = 100.0 * xt::ones<double>({N});
//     std::vector<int> direction;
//     std::vector<xt::xtensor<size_t, 1>> particles;

//     for (size_t i = 0; i < n; ++i) {
//         bool r = sys.set_x(xt::eval((double)(i) * x));
//         if (r) {
//             auto iredraw = sys.getRedrawList().currentRedraw();
//             xt::xtensor<size_t, 1> r = xt::flatten_indices(xt::argwhere(iredraw > 0));
//             xt::xtensor<size_t, 1> l = xt::flatten_indices(xt::argwhere(iredraw < 0));
//             if (r.size() > 0) {
//                 direction.push_back(1);
//                 particles.push_back(r);
//             }
//             if (l.size() > 0) {
//                 direction.push_back(-1);
//                 particles.push_back(l);
//             }
//         }
//     }

//     QPot::random::seed(seed);
//     FrictionQPotSpringBlock::Line1d::System other(N, random);

//     for (size_t i = 0; i < direction.size(); ++i) {
//         if (direction[i] > 0) {
//             other.getRedrawList().redrawRight(particles[i]);
//         }
//         else {
//             other.getRedrawList().redrawLeft(particles[i]);
//         }
//     }

//     other.set_x(sys.x());

//     REQUIRE(xt::allclose(sys.yieldLeft(), other.yieldLeft()));
// }


}
