
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotSpringBlock/Line1d.h>
#include <QPot/random.hpp>

TEST_CASE("FrictionQPotSpringBlock::Line1d", "Line1d.h")
{

SECTION("System::advanceRightElastic")
{
    size_t N = 3;
    auto uniform = [=](std::array<size_t, 2> shape) {
        return xt::ones<double>(shape); };

    FrictionQPotSpringBlock::Line1d::System sys(N, uniform);

    sys.set_dt(0.1);
    sys.set_eta(2.0 * std::sqrt(3.0) / 10.0);
    sys.set_m(1.0);
    sys.set_mu(1.0);
    sys.set_k_neighbours(1.0);
    sys.set_k_frame(1.0 / double(N));
    sys.set_x_frame(0.0);

    sys.advanceRightElastic(0.2);

    xt::xtensor<double, 1> x = (0.5 - 0.1) * xt::ones<double>({N});

    REQUIRE(xt::allclose(x, sys.x()));
}

SECTION("Reconstruct sequence")
{
    size_t N = 3;
    auto seed = time(NULL);
    auto random = QPot::random::RandList();

    QPot::random::seed(seed);
    FrictionQPotSpringBlock::Line1d::System sys(N, random);

    size_t n = 20;
    xt::xtensor<double, 1> x = 100.0 * xt::ones<double>({N});
    std::vector<xt::xtensor<int, 1>> redraw;

    for (size_t i = 0; i < n; ++i) {
        bool r = sys.set_x(xt::eval((double)(i) * x));
        if (r) {
            redraw.push_back(sys.QPot().currentRedraw());
        }
    }

    QPot::random::seed(seed);
    FrictionQPotSpringBlock::Line1d::System other(N, random);

    for (auto& i : redraw) {
        other.QPot().redraw(i);
    }

    other.set_x(sys.x());

    REQUIRE(xt::allclose(sys.yieldLeft(), other.yieldLeft()));

}

}
