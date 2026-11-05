#pragma once

#include <vector>

namespace fpinn {

constexpr double kPi = 3.14159265358979323846;

/// Standard normal cumulative distribution function N(x).
double norm_cdf(double x);

/// Standard normal probability density function n(x).
double norm_pdf(double x);

/// One-parameter Mittag-Leffler function
///     E_alpha(z) = sum_{k>=0} z^k / Gamma(alpha k + 1),   0 < alpha <= 1.
///
/// It is the "fractional exponential": E_1(z) = exp(z).  In the time-fractional
/// Black-Scholes model it replaces the discount factor exp(-r tau) by
/// E_alpha(-r tau^alpha).
///
/// Evaluated by its power series, which is accurate as long as |z|^(1/alpha)
/// is moderate (< ~18, i.e. far beyond anything an interest rate produces).
/// Throws std::invalid_argument / std::domain_error outside that range.
double mittag_leffler(double alpha, double z);

/// Gauss quadrature rule for the "random clock" of the time-fractional model.
///
/// The solution of  D^alpha v = A v,  v(0) = f  is  v(tau) = E[ e^{tau^alpha Y A} f ],
/// i.e. the classical solution evaluated at the random time tau^alpha * Y, where Y
/// has the M-Wright density M_alpha on (0, inf) with moments
///     E[Y^k] = k! / Gamma(1 + alpha k).
/// This returns nodes y_q and weights w_q (q = 1..n) with
///     E[g(Y)] ~ sum_q w_q g(y_q),
/// exact for polynomials of degree <= 2n-1 (computed from the moments with
/// Gautschi's Chebyshev algorithm and the Golub-Welsch eigenvalue method).
/// For alpha = 1, Y = 1 and a single node is returned.  n is reduced
/// automatically if the moment problem becomes numerically ill-conditioned.
void random_clock_gauss_rule(double alpha, int n, std::vector<double>& nodes, std::vector<double>& weights);

}  // namespace fpinn
