#pragma once

#include "fpinn/problem.hpp"

namespace fpinn {

/// Closed-form classical Black-Scholes results (valid only for alpha = 1).
/// Used to validate the solvers.
struct BSResult {
    double price;
    double delta;  ///< dV/dS
    double gamma;  ///< d2V/dS2
};

BSResult black_scholes(OptionType type, double S, double K, double r, double sigma, double tau);

/// Black-Scholes price in dimensionless variables (x = ln(S/K), v = V/K) together
/// with its first and second x-derivatives.  theta is the time to maturity.
struct BSJet {
    double v, vx, vxx;
};
BSJet black_scholes_x(OptionType type, double x, double r, double sigma, double theta);

/// Exact price of the time-fractional model for alpha = 1/2 via subordination:
///
///   V_alpha(S, tau) = integral_0^inf V_BS(S, s) h_alpha(s; tau) ds,
///
/// where h_alpha is the density of the "random clock" (inverse alpha-stable
/// subordinator).  For alpha = 1/2 that density is a half-Gaussian,
///   h(s; tau) = exp(-s^2 / (4 tau)) / sqrt(pi tau).
/// Used as an independent check of the numerical solvers.
double fractional_bs_alpha_half(OptionType type, double S, double K, double r, double sigma, double tau);

}  // namespace fpinn
