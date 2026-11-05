#include <cmath>
#include <vector>

#include "check.hpp"
#include "fpinn/caputo.hpp"

using namespace fpinn;

namespace {
// Error of the L1 scheme for u = tau^2 at tau = T, whose exact Caputo
// derivative is 2 tau^(2-alpha) / Gamma(3-alpha).
double l1_error_tau2(double alpha, int M, double grading) {
    const double T = 1.0;
    L1Operator op(alpha, make_time_grid(T, M, grading));
    std::vector<double> u(M + 1);
    for (int j = 0; j <= M; ++j) u[j] = op.grid()[j] * op.grid()[j];
    const double exact = 2.0 * std::pow(T, 2.0 - alpha) / std::tgamma(3.0 - alpha);
    return std::abs(op.apply(M, u.data()) - exact);
}
}  // namespace

int main() {
    // Grid.
    const auto g = make_time_grid(2.0, 4, 2.0);
    CHECK_NEAR(g[0], 0.0, 0.0);
    CHECK_NEAR(g[2], 0.5, 1e-15);
    CHECK_NEAR(g[4], 2.0, 0.0);

    // alpha = 1 is the backward difference.
    {
        L1Operator op(1.0, make_time_grid(1.0, 10, 1.5));
        for (int n = 1; n <= 10; ++n) {
            const double dt = op.grid()[n] - op.grid()[n - 1];
            CHECK_NEAR(op.weight(n, n), 1.0 / dt, 1e-12);
            CHECK_NEAR(op.weight(n, n - 1), -1.0 / dt, 1e-12);
            for (int j = 0; j < n - 1; ++j) CHECK_NEAR(op.weight(n, j), 0.0, 0.0);
        }
    }

    for (double alpha : {0.3, 0.5, 0.7, 0.9}) {
        for (double grading : {1.0, 2.0}) {
            L1Operator op(alpha, make_time_grid(1.0, 25, grading));
            const auto& t = op.grid();
            // Derivative of a constant is zero: every row sums to 0.
            // Linear functions are differentiated exactly: D^a tau = tau^(1-a)/Gamma(2-a).
            std::vector<double> u(t.begin(), t.end());
            for (int n = 1; n <= op.steps(); ++n) {
                double s = 0.0;
                for (double w : op.row(n)) s += w;
                CHECK_NEAR(s, 0.0, 1e-9 * std::abs(op.weight(n, n)));
                CHECK_NEAR(op.apply(n, u.data()), std::pow(t[n], 1.0 - alpha) / std::tgamma(2.0 - alpha), 1e-11);
            }
        }
        // Convergence of order 2 - alpha on a uniform grid for a smooth function.
        const double e1 = l1_error_tau2(alpha, 50, 1.0);
        const double e2 = l1_error_tau2(alpha, 100, 1.0);
        const double order = std::log2(e1 / e2);
        CHECK_NEAR(order, 2.0 - alpha, 0.1);
        CHECK(e2 < 1e-2);  // absolute error at M = 100 (largest for alpha near 1)
    }

    return test_summary("test_caputo");
}
