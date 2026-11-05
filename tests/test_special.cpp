#include <cmath>
#include <stdexcept>

#include "check.hpp"
#include "fpinn/black_scholes.hpp"
#include "fpinn/special_functions.hpp"

using namespace fpinn;

int main() {
    // Normal distribution.
    CHECK_NEAR(norm_cdf(0.0), 0.5, 1e-15);
    CHECK_NEAR(norm_cdf(1.96), 0.9750021048517795, 1e-13);
    CHECK_NEAR(norm_cdf(-1.0) + norm_cdf(1.0), 1.0, 1e-15);
    CHECK_NEAR(norm_pdf(0.0), 1.0 / std::sqrt(2.0 * kPi), 1e-15);

    // Mittag-Leffler: E_alpha(0) = 1, E_1(z) = exp(z).
    for (double a : {0.3, 0.5, 0.8, 1.0}) CHECK_NEAR(mittag_leffler(a, 0.0), 1.0, 0.0);
    for (double z : {-3.0, -0.5, 0.0, 0.7, 2.0}) CHECK_NEAR(mittag_leffler(1.0, z), std::exp(z), 1e-14);

    // E_{1/2}(-z) = exp(z^2) erfc(z).
    for (double z : {0.01, 0.1, 0.5, 1.0, 2.0}) {
        CHECK_NEAR(mittag_leffler(0.5, -z), std::exp(z * z) * std::erfc(z), 1e-12);
        CHECK_NEAR(mittag_leffler(0.5, z), std::exp(z * z) * std::erfc(-z), 1e-12 * std::exp(z * z));
    }

    // Continuity in alpha near 1.
    CHECK_NEAR(mittag_leffler(0.999999, -0.05), std::exp(-0.05), 1e-6);

    // Invalid arguments.
    bool threw = false;
    try {
        mittag_leffler(1.5, 0.1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        mittag_leffler(0.5, -100.0);
    } catch (const std::domain_error&) {
        threw = true;
    }
    CHECK(threw);

    // Random-clock Gauss rule: weights sum to one, moments k!/Gamma(1+alpha k) are
    // reproduced up to degree 2n-1, and its Laplace transform is E_alpha(-s).
    for (double a : {0.3, 0.5, 0.8, 0.95}) {
        for (int n : {1, 2, 3, 4}) {
            std::vector<double> y, w;
            random_clock_gauss_rule(a, n, y, w);
            CHECK(static_cast<int>(y.size()) == n);
            for (std::size_t q = 0; q < y.size(); ++q) CHECK(y[q] > 0.0 && w[q] > 0.0);
            for (int k = 0; k <= 2 * n - 1; ++k) {
                double m = 0.0;
                for (std::size_t q = 0; q < y.size(); ++q) m += w[q] * std::pow(y[q], k);
                const double exact = std::tgamma(k + 1.0) / std::tgamma(1.0 + a * k);
                CHECK_NEAR(m, exact, 1e-10 * exact);
            }
        }
        std::vector<double> y, w;
        random_clock_gauss_rule(a, 6, y, w);
        for (double s : {0.05, 0.5, 1.0}) {
            double lt = 0.0;
            for (std::size_t q = 0; q < y.size(); ++q) lt += w[q] * std::exp(-s * y[q]);
            CHECK_NEAR(lt, mittag_leffler(a, -s), 1e-6);
        }
    }
    {
        std::vector<double> y, w;
        random_clock_gauss_rule(1.0, 3, y, w);  // alpha = 1: deterministic clock
        CHECK(y.size() == 1 && y[0] == 1.0 && w[0] == 1.0);
    }

    // Black-Scholes in x = ln(S/K): consistent with the market-variable formula,
    // and its x-derivatives agree with finite differences.
    for (OptionType t : {OptionType::Call, OptionType::Put}) {
        const double K = 100.0, r = 0.05, sig = 0.2, th = 0.7;
        for (double S : {70.0, 100.0, 135.0}) {
            const double x = std::log(S / K);
            const BSJet j = black_scholes_x(t, x, r, sig, th);
            const BSResult m = black_scholes(t, S, K, r, sig, th);
            CHECK_NEAR(K * j.v, m.price, 1e-11);
            CHECK_NEAR(j.vx * K / S, m.delta, 1e-12);                  // dV/dS = (K v_x) / S
            CHECK_NEAR((j.vxx - j.vx) * K / (S * S), m.gamma, 1e-12);  // d2V/dS2
            const double h = 1e-4;
            const double vp = black_scholes_x(t, x + h, r, sig, th).v, vm = black_scholes_x(t, x - h, r, sig, th).v;
            CHECK_NEAR(j.vx, (vp - vm) / (2 * h), 1e-8);
            CHECK_NEAR(j.vxx, (vp - 2 * j.v + vm) / (h * h), 1e-5);
        }
    }

    return test_summary("test_special");
}
