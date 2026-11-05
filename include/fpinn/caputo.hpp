#pragma once

#include <vector>

namespace fpinn {

/// Time grid 0 = tau_0 < tau_1 < ... < tau_M = T with
///     tau_j = T * (j / M)^grading.
/// grading = 1 gives a uniform grid; grading > 1 clusters points near tau = 0,
/// where the solution of a fractional equation changes fastest.
std::vector<double> make_time_grid(double T, int M, double grading);

/// L1 discretisation of the Caputo fractional derivative of order alpha in (0, 1]
/// on an arbitrary (possibly non-uniform) grid:
///
///   D^alpha u(tau_n) ~ 1/Gamma(2-alpha) * sum_{j=1..n} (u_j - u_{j-1}) / dtau_j
///                       * [ (tau_n - tau_{j-1})^(1-alpha) - (tau_n - tau_j)^(1-alpha) ]
///
/// The operator is stored as a lower-triangular matrix W so that
///   D^alpha u(tau_n) ~ sum_{j=0..n} W(n, j) * u_j        (n >= 1).
/// For alpha = 1 it reduces to the backward difference (u_n - u_{n-1}) / dtau_n.
class L1Operator {
public:
    L1Operator(double alpha, std::vector<double> tau);

    double alpha() const { return alpha_; }
    int steps() const { return static_cast<int>(tau_.size()) - 1; }
    const std::vector<double>& grid() const { return tau_; }

    /// Weights W(n, 0..n) of row n (n >= 1).
    const std::vector<double>& row(int n) const { return rows_[n]; }
    double weight(int n, int j) const { return rows_[n][j]; }

    /// Apply row n to the history u[0..n].
    double apply(int n, const double* u) const;

private:
    double alpha_;
    std::vector<double> tau_;
    std::vector<std::vector<double>> rows_;
};

}  // namespace fpinn
