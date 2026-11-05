#include "fpinn/caputo.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace fpinn {

std::vector<double> make_time_grid(double T, int M, double grading) {
    if (T <= 0.0 || M < 1 || grading < 1.0) {
        throw std::invalid_argument("make_time_grid: need T > 0, M >= 1, grading >= 1");
    }
    std::vector<double> tau(M + 1);
    for (int j = 0; j <= M; ++j) {
        tau[j] = T * std::pow(static_cast<double>(j) / M, grading);
    }
    tau[M] = T;  // exact end point, no rounding
    return tau;
}

L1Operator::L1Operator(double alpha, std::vector<double> tau) : alpha_(alpha), tau_(std::move(tau)) {
    if (!(alpha_ > 0.0 && alpha_ <= 1.0)) {
        throw std::invalid_argument("L1Operator: alpha must be in (0, 1]");
    }
    if (tau_.size() < 2) throw std::invalid_argument("L1Operator: need at least two grid points");
    for (std::size_t j = 1; j < tau_.size(); ++j) {
        if (!(tau_[j] > tau_[j - 1])) throw std::invalid_argument("L1Operator: grid must increase");
    }

    const int M = steps();
    rows_.assign(M + 1, {});
    const double p = 1.0 - alpha_;
    const double inv_gamma = 1.0 / std::tgamma(2.0 - alpha_);

    for (int n = 1; n <= M; ++n) {
        std::vector<double>& w = rows_[n];
        w.assign(n + 1, 0.0);
        for (int j = 1; j <= n; ++j) {
            const double dt = tau_[j] - tau_[j - 1];
            double a;
            if (alpha_ == 1.0) {
                // Classical derivative: only the last interval contributes.
                a = (j == n) ? 1.0 / dt : 0.0;
            } else {
                const double hi = std::pow(tau_[n] - tau_[j - 1], p);
                const double lo = (j == n) ? 0.0 : std::pow(tau_[n] - tau_[j], p);
                a = inv_gamma * (hi - lo) / dt;
            }
            w[j] += a;
            w[j - 1] -= a;
        }
    }
}

double L1Operator::apply(int n, const double* u) const {
    const std::vector<double>& w = rows_[n];
    double s = 0.0;
    for (int j = 0; j <= n; ++j) s += w[j] * u[j];
    return s;
}

}  // namespace fpinn
