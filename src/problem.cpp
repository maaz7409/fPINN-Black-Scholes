#include "fpinn/problem.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "fpinn/special_functions.hpp"

namespace fpinn {

OptionType parse_option_type(const std::string& s) {
    if (s == "call" || s == "Call" || s == "CALL") return OptionType::Call;
    if (s == "put" || s == "Put" || s == "PUT") return OptionType::Put;
    throw std::invalid_argument("unknown option type '" + s + "' (use call or put)");
}

const char* to_string(OptionType t) { return t == OptionType::Call ? "call" : "put"; }

void FractionalBS::validate() const {
    if (!(K > 0.0)) throw std::invalid_argument("K must be > 0");
    if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");
    if (!(T > 0.0)) throw std::invalid_argument("T must be > 0");
    if (!(alpha > 0.0 && alpha <= 1.0)) throw std::invalid_argument("alpha must be in (0, 1]");
    if (!(L > 0.0)) throw std::invalid_argument("L must be > 0");
    if (r < 0.0) throw std::invalid_argument("r must be >= 0");
}

double FractionalBS::discount(double tau) const {
    if (tau <= 0.0) return 1.0;
    return mittag_leffler(alpha, -r * std::pow(tau, alpha));
}

double FractionalBS::payoff(double x) const {
    return type == OptionType::Call ? std::max(std::exp(x) - 1.0, 0.0) : std::max(1.0 - std::exp(x), 0.0);
}

// Far-field behaviour.  e^x solves the spatial operator exactly (a + b - r = 0)
// and g(tau) = E_alpha(-r tau^alpha) solves D^alpha g = -r g, g(0) = 1.  Hence
//   call:  v ~ e^x - g(tau)  for x -> +inf,   v ~ 0 for x -> -inf,
//   put:   v ~ g(tau) - e^x  for x -> -inf,   v ~ 0 for x -> +inf.
double FractionalBS::bc_left(double tau) const {
    return type == OptionType::Call ? 0.0 : discount(tau) - std::exp(-L);
}

double FractionalBS::bc_right(double tau) const {
    return type == OptionType::Call ? std::exp(L) - discount(tau) : 0.0;
}

double FractionalBS::x_of(double S) const { return std::log(S / K); }
double FractionalBS::S_of(double x) const { return K * std::exp(x); }

}  // namespace fpinn
