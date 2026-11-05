#pragma once

#include <vector>

#include "fpinn/caputo.hpp"
#include "fpinn/problem.hpp"

namespace fpinn {

struct FDConfig {
    int nx = 1200;         ///< number of spatial intervals on [-L, L]
    int nt = 1000;         ///< number of time steps on [0, T]
    double grading = 2.0;  ///< time-grid grading (see make_time_grid)
};

/// Classical reference solver for the time-fractional Black-Scholes equation:
///   * L1 scheme for the Caputo derivative in time (implicit),
///   * second-order central differences in x,
///   * one tridiagonal (Thomas) solve per time step.
/// Cost is O(nt^2 * nx) because the fractional derivative has full memory.
/// It is used as the "ground truth" against which the fPINN is measured.
class FDSolver {
public:
    FDSolver(const FractionalBS& problem, FDConfig cfg = {});

    void solve();

    const std::vector<double>& x_grid() const { return x_; }
    const std::vector<double>& tau_grid() const { return op_.grid(); }
    /// Solution v(x_i, tau_n) for all i, at time level n.
    const std::vector<double>& level(int n) const { return u_[n]; }

    /// v, v_x, v_xx at (x, tau) using cubic Lagrange interpolation in x and
    /// linear interpolation in tau.
    struct Jet {
        double v, vx, vxx;
    };
    Jet eval(double x, double tau) const;

    /// Market-variable outputs.
    double price(double S, double tau) const;
    double delta(double S, double tau) const;
    double gamma(double S, double tau) const;

private:
    Jet eval_level(double x, int n) const;

    FractionalBS p_;
    FDConfig cfg_;
    std::vector<double> x_;
    L1Operator op_;
    std::vector<std::vector<double>> u_;
    bool solved_ = false;
};

}  // namespace fpinn
