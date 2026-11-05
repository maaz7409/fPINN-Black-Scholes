#include "fpinn/fd_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fpinn {

FDSolver::FDSolver(const FractionalBS& problem, FDConfig cfg)
    : p_(problem), cfg_(cfg), op_(problem.alpha, make_time_grid(problem.T, cfg.nt, cfg.grading)) {
    p_.validate();
    if (cfg_.nx < 4) throw std::invalid_argument("FDSolver: nx must be >= 4");
    x_.resize(cfg_.nx + 1);
    const double h = 2.0 * p_.L / cfg_.nx;
    for (int i = 0; i <= cfg_.nx; ++i) x_[i] = -p_.L + i * h;
    x_[cfg_.nx] = p_.L;
}

void FDSolver::solve() {
    const int N = cfg_.nx;
    const int M = op_.steps();
    const double h = 2.0 * p_.L / N;
    const double a = p_.diffusion();
    const double b = p_.drift();
    const double r = p_.r;
    const std::vector<double>& tau = op_.grid();

    u_.assign(M + 1, std::vector<double>(N + 1, 0.0));
    for (int i = 0; i <= N; ++i) u_[0][i] = p_.payoff(x_[i]);

    // Tridiagonal coefficients of  -L_h  (constant in time).
    const double lo = -(a / (h * h) - b / (2.0 * h));  // multiplies u_{i-1}
    const double up = -(a / (h * h) + b / (2.0 * h));  // multiplies u_{i+1}
    const double mid = 2.0 * a / (h * h) + r;          // multiplies u_i

    std::vector<double> rhs(N + 1), cp(N + 1), dp(N + 1);
    for (int n = 1; n <= M; ++n) {
        const std::vector<double>& w = op_.row(n);
        // Memory term: -sum_{j<n} W(n,j) u_j  (moved to the right-hand side).
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (int j = 0; j < n; ++j) {
            const double wj = w[j];
            const double* uj = u_[j].data();
            for (int i = 0; i <= N; ++i) rhs[i] -= wj * uj[i];
        }

        std::vector<double>& un = u_[n];
        un[0] = p_.bc_left(tau[n]);
        un[N] = p_.bc_right(tau[n]);

        // Solve  (W(n,n) - L_h) u_n = rhs  for the interior nodes 1..N-1.
        const double diag = w[n] + mid;
        rhs[1] -= lo * un[0];
        rhs[N - 1] -= up * un[N];
        // Thomas algorithm.
        cp[1] = up / diag;
        dp[1] = rhs[1] / diag;
        for (int i = 2; i <= N - 1; ++i) {
            const double m = diag - lo * cp[i - 1];
            cp[i] = up / m;
            dp[i] = (rhs[i] - lo * dp[i - 1]) / m;
        }
        un[N - 1] = dp[N - 1];
        for (int i = N - 2; i >= 1; --i) un[i] = dp[i] - cp[i] * un[i + 1];
    }
    solved_ = true;
}

FDSolver::Jet FDSolver::eval_level(double x, int n) const {
    const int N = cfg_.nx;
    const double h = 2.0 * p_.L / N;
    int k = static_cast<int>(std::floor((x + p_.L) / h));
    k = std::clamp(k, 1, N - 2);  // stencil k-1 .. k+2 must exist
    const double s = (x - x_[k]) / h;

    // Cubic Lagrange basis on nodes s = -1, 0, 1, 2 and its derivatives.
    const double l[4] = {-s * (s - 1.0) * (s - 2.0) / 6.0, (s + 1.0) * (s - 1.0) * (s - 2.0) / 2.0,
                         -(s + 1.0) * s * (s - 2.0) / 2.0, (s + 1.0) * s * (s - 1.0) / 6.0};
    const double d1[4] = {-(3.0 * s * s - 6.0 * s + 2.0) / 6.0, (3.0 * s * s - 4.0 * s - 1.0) / 2.0,
                          -(3.0 * s * s - 2.0 * s - 2.0) / 2.0, (3.0 * s * s - 1.0) / 6.0};
    const double d2[4] = {-(s - 1.0), 3.0 * s - 2.0, -3.0 * s + 1.0, s};

    const std::vector<double>& u = u_[n];
    Jet j{0.0, 0.0, 0.0};
    for (int m = 0; m < 4; ++m) {
        const double um = u[k - 1 + m];
        j.v += l[m] * um;
        j.vx += d1[m] * um;
        j.vxx += d2[m] * um;
    }
    j.vx /= h;
    j.vxx /= h * h;
    return j;
}

FDSolver::Jet FDSolver::eval(double x, double tau) const {
    if (!solved_) throw std::logic_error("FDSolver::eval called before solve()");
    const std::vector<double>& t = op_.grid();
    if (tau <= 0.0) return eval_level(x, 0);
    if (tau >= t.back()) return eval_level(x, op_.steps());
    const auto it = std::upper_bound(t.begin(), t.end(), tau);
    const int n1 = static_cast<int>(it - t.begin());
    const int n0 = n1 - 1;
    const double w = (tau - t[n0]) / (t[n1] - t[n0]);
    const Jet a = eval_level(x, n0);
    const Jet b = eval_level(x, n1);
    return {(1 - w) * a.v + w * b.v, (1 - w) * a.vx + w * b.vx, (1 - w) * a.vxx + w * b.vxx};
}

double FDSolver::price(double S, double tau) const { return p_.K * eval(p_.x_of(S), tau).v; }

double FDSolver::delta(double S, double tau) const {
    const double x = p_.x_of(S);
    return eval(x, tau).vx * std::exp(-x);
}

double FDSolver::gamma(double S, double tau) const {
    const double x = p_.x_of(S);
    const Jet j = eval(x, tau);
    return (j.vxx - j.vx) * std::exp(-2.0 * x) / p_.K;
}

}  // namespace fpinn
