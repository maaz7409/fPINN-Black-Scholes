#pragma once

#include <string>

namespace fpinn {

enum class OptionType { Call, Put };

OptionType parse_option_type(const std::string& s);
const char* to_string(OptionType t);

/// Time-fractional Black-Scholes model (Wyss 2000) for a European option.
///
/// With tau = T - t (time to maturity) the price V(S, tau) satisfies
///
///   D_tau^alpha V = 1/2 sigma^2 S^2 V_SS + r S V_S - r V,   V(S, 0) = payoff(S),
///
/// where D_tau^alpha is the Caputo derivative of order alpha in (0, 1].
/// alpha = 1 is the classical Black-Scholes equation.
///
/// Internally we work in dimensionless variables
///     x = ln(S / K),   v(x, tau) = V / K,
/// which turn the equation into one with constant coefficients:
///
///   D_tau^alpha v = a v_xx + b v_x - r v,   a = sigma^2/2,  b = r - sigma^2/2,
///
/// solved on the truncated domain x in [-L, L].
struct FractionalBS {
    double K = 100.0;      ///< strike
    double r = 0.05;       ///< risk-free rate
    double sigma = 0.2;    ///< volatility
    double T = 1.0;        ///< maturity (years)
    double alpha = 1.0;    ///< fractional order, 0 < alpha <= 1
    double L = 1.5;        ///< domain half-width in log-moneyness (S in [K e^-L, K e^L])
    OptionType type = OptionType::Call;

    void validate() const;  ///< throws std::invalid_argument on bad parameters

    double diffusion() const { return 0.5 * sigma * sigma; }  ///< a
    double drift() const { return r - 0.5 * sigma * sigma; }  ///< b

    /// "Fractional discount factor" E_alpha(-r tau^alpha); equals exp(-r tau) for alpha = 1.
    double discount(double tau) const;

    double payoff(double x) const;  ///< v(x, 0)
    double bc_left(double tau) const;   ///< v(-L, tau)
    double bc_right(double tau) const;  ///< v(+L, tau)

    /// Spatial operator applied to given derivatives: a v_xx + b v_x - r v.
    double spatial_operator(double v, double vx, double vxx) const {
        return diffusion() * vxx + drift() * vx - r * v;
    }

    // Conversions between market and dimensionless variables.
    double x_of(double S) const;
    double S_of(double x) const;
};

}  // namespace fpinn
