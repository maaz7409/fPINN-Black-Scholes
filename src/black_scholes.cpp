#include "fpinn/black_scholes.hpp"

#include <algorithm>
#include <cmath>

#include "fpinn/special_functions.hpp"

namespace fpinn {

BSResult black_scholes(OptionType type, double S, double K, double r, double sigma, double tau) {
    if (tau <= 0.0) {
        const bool call = type == OptionType::Call;
        const double pay = call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        const double delta = call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0);
        return {pay, delta, 0.0};
    }
    const double sq = sigma * std::sqrt(tau);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * tau) / sq;
    const double d2 = d1 - sq;
    const double df = std::exp(-r * tau);
    const double gamma = norm_pdf(d1) / (S * sq);
    if (type == OptionType::Call) {
        return {S * norm_cdf(d1) - K * df * norm_cdf(d2), norm_cdf(d1), gamma};
    }
    return {K * df * norm_cdf(-d2) - S * norm_cdf(-d1), norm_cdf(d1) - 1.0, gamma};
}

BSJet black_scholes_x(OptionType type, double x, double r, double sigma, double theta) {
    const double ex = std::exp(x);
    const bool call = type == OptionType::Call;
    if (theta <= 0.0) {
        const bool itm = call ? x > 0.0 : x < 0.0;
        if (!itm) return {0.0, 0.0, 0.0};
        return call ? BSJet{ex - 1.0, ex, ex} : BSJet{1.0 - ex, -ex, -ex};
    }
    const double sq = sigma * std::sqrt(theta);
    const double d1 = (x + (r + 0.5 * sigma * sigma) * theta) / sq;
    const double d2 = d1 - sq;
    const double df = std::exp(-r * theta);
    const double curv = ex * norm_pdf(d1) / sq;  // S^2 V_SS / K
    if (call) {
        const double vx = ex * norm_cdf(d1);
        return {vx - df * norm_cdf(d2), vx, vx + curv};
    }
    const double vx = -ex * norm_cdf(-d1);
    return {df * norm_cdf(-d2) + vx, vx, vx + curv};
}

double fractional_bs_alpha_half(OptionType type, double S, double K, double r, double sigma, double tau) {
    if (tau <= 0.0) return black_scholes(type, S, K, r, sigma, 0.0).price;
    // Substitute s = sqrt(tau) * u^2 (removes the sqrt(s) behaviour of V_BS at s = 0):
    //   integral = int_0^inf V_BS(S, sqrt(tau) u^2) * 2u exp(-u^4/4) / sqrt(pi) du.
    // exp(-u^4/4) < 1e-30 for u > 5.1, so integrate u in [0, 6] with composite Simpson.
    const int n = 4000;  // even
    const double umax = 6.0, h = umax / n, st = std::sqrt(tau);
    double sum = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double u = i * h;
        const double wgt = (i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
        const double f = black_scholes(type, S, K, r, sigma, st * u * u).price * 2.0 * u *
                         std::exp(-0.25 * u * u * u * u) / std::sqrt(kPi);
        sum += wgt * f;
    }
    return sum * h / 3.0;
}

}  // namespace fpinn
