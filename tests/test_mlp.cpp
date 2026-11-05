#include <random>
#include <sstream>
#include <vector>

#include "check.hpp"
#include "fpinn/mlp.hpp"

using namespace fpinn;

int main() {
    MLP net({2, 8, 7, 1});
    net.init_xavier(7);
    // Non-zero biases so every code path is exercised.
    std::mt19937_64 rng(3);
    std::normal_distribution<double> nd(0.0, 0.5);
    for (double& p : net.params()) p += 0.3 * nd(rng);

    auto ws = net.make_workspace();
    const double z0[2] = {0.3, -0.4};
    const double dir[2] = {0.7, -0.3};

    // eval() agrees with forward().v
    const Jet j = net.forward(z0, dir, ws);
    CHECK_NEAR(net.eval(z0, ws), j.v, 1e-15);

    // Directional derivatives against finite differences.
    auto f = [&](double s) {
        const double z[2] = {z0[0] + s * dir[0], z0[1] + s * dir[1]};
        return net.eval(z, ws);
    };
    const double h = 1e-4;
    CHECK_NEAR(j.d, (f(h) - f(-h)) / (2 * h), 1e-8);
    CHECK_NEAR(j.dd, (f(h) - 2 * f(0) + f(-h)) / (h * h), 1e-5);

    // Parameter gradient of J = c0 N + c1 N' + c2 N'' against finite differences.
    const Jet seed{0.3, -1.1, 0.7};
    std::vector<double> grad(net.num_params(), 0.0);
    net.forward(z0, dir, ws);
    net.backward(seed, ws, grad.data());
    auto J = [&]() {
        const Jet q = net.forward(z0, dir, ws);
        return seed.v * q.v + seed.d * q.d + seed.dd * q.dd;
    };
    double max_err = 0.0, max_g = 0.0;
    for (std::size_t k = 0; k < net.num_params(); ++k) {
        const double p = net.params()[k];
        const double eps = 1e-6;
        net.params()[k] = p + eps;
        const double jp = J();
        net.params()[k] = p - eps;
        const double jm = J();
        net.params()[k] = p;
        const double fd = (jp - jm) / (2 * eps);
        max_err = std::max(max_err, std::abs(fd - grad[k]));
        max_g = std::max(max_g, std::abs(grad[k]));
    }
    CHECK(max_g > 1e-3);
    CHECK_NEAR(max_err / max_g, 0.0, 1e-7);

    // Save / load round trip.
    std::stringstream ss;
    net.save(ss);
    MLP back = MLP::load(ss);
    CHECK(back.layer_sizes() == net.layer_sizes());
    auto ws2 = back.make_workspace();
    CHECK_NEAR(back.eval(z0, ws2), net.eval(z0, ws), 0.0);

    return test_summary("test_mlp");
}
