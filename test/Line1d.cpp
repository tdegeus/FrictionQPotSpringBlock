
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotSpringBlock/Line1d.h>

TEST_CASE("FrictionQPotSpringBlock::Line1d", "Line1d.h")
{

SECTION("System::advanceRightElastic")
{
    size_t N = 3;
    auto uniform = [=](std::vector<size_t> shape) {
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

    xt::xtensor<double, 1> x = 0.9 * xt::ones<double>({N});

    REQUIRE(xt::allclose(x, sys.get_x()));
}

}
