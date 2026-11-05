#include "fpinn/optim.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace fpinn {

Adam::Adam(std::size_t n, double beta1, double beta2, double eps)
    : b1_(beta1), b2_(beta2), eps_(eps), m_(n, 0.0), v_(n, 0.0) {}

void Adam::step(std::vector<double>& params, const std::vector<double>& grad, double lr) {
    if (params.size() != m_.size() || grad.size() != m_.size()) {
        throw std::invalid_argument("Adam::step: size mismatch");
    }
    ++t_;
    const double c1 = 1.0 - std::pow(b1_, static_cast<double>(t_));
    const double c2 = 1.0 - std::pow(b2_, static_cast<double>(t_));
    for (std::size_t i = 0; i < params.size(); ++i) {
        m_[i] = b1_ * m_[i] + (1.0 - b1_) * grad[i];
        v_[i] = b2_ * v_[i] + (1.0 - b2_) * grad[i] * grad[i];
        params[i] -= lr * (m_[i] / c1) / (std::sqrt(v_[i] / c2) + eps_);
    }
}

namespace {
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
double max_abs(const std::vector<double>& a) {
    double m = 0.0;
    for (double v : a) m = std::max(m, std::abs(v));
    return m;
}
}  // namespace

int lbfgs_minimize(const Objective& f, std::vector<double>& x, const LBFGSConfig& cfg,
                   const std::function<void(int, double)>& on_iter) {
    const std::size_t n = x.size();
    std::vector<double> g(n), g_new(n), x_new(n), d(n);
    std::deque<std::vector<double>> S, Y;
    std::deque<double> rho;

    double fx = f(x, g);
    int iter = 0;
    for (; iter < cfg.max_iter; ++iter) {
        if (!std::isfinite(fx) || max_abs(g) < cfg.grad_tol) break;

        // Two-loop recursion: d = -H g.
        d = g;
        std::vector<double> alpha(S.size());
        for (int k = static_cast<int>(S.size()) - 1; k >= 0; --k) {
            alpha[k] = rho[k] * dot(S[k], d);
            for (std::size_t i = 0; i < n; ++i) d[i] -= alpha[k] * Y[k][i];
        }
        const double gamma = S.empty() ? 1.0 / std::max(1.0, max_abs(g))
                                       : dot(S.back(), Y.back()) / dot(Y.back(), Y.back());
        for (double& v : d) v *= gamma;
        for (std::size_t k = 0; k < S.size(); ++k) {
            const double beta = rho[k] * dot(Y[k], d);
            for (std::size_t i = 0; i < n; ++i) d[i] += S[k][i] * (alpha[k] - beta);
        }
        for (double& v : d) v = -v;

        double slope = dot(g, d);
        if (!(slope < 0.0)) {  // not a descent direction: restart with steepest descent
            S.clear();
            Y.clear();
            rho.clear();
            for (std::size_t i = 0; i < n; ++i) d[i] = -g[i] / std::max(1.0, max_abs(g));
            slope = dot(g, d);
        }

        // Armijo back-tracking line search.
        double t = 1.0, f_new = 0.0;
        bool accepted = false;
        for (int ls = 0; ls < 40; ++ls) {
            for (std::size_t i = 0; i < n; ++i) x_new[i] = x[i] + t * d[i];
            f_new = f(x_new, g_new);
            if (std::isfinite(f_new) && f_new <= fx + 1e-4 * t * slope) {
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if (!accepted) break;

        std::vector<double> s(n), y(n);
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = x_new[i] - x[i];
            y[i] = g_new[i] - g[i];
        }
        const double sy = dot(s, y);
        if (sy > 1e-12 * std::sqrt(dot(s, s) * dot(y, y))) {
            S.push_back(std::move(s));
            Y.push_back(std::move(y));
            rho.push_back(1.0 / sy);
            if (static_cast<int>(S.size()) > cfg.history) {
                S.pop_front();
                Y.pop_front();
                rho.pop_front();
            }
        }
        x.swap(x_new);
        g.swap(g_new);
        fx = f_new;
        if (on_iter) on_iter(iter + 1, fx);
    }
    return iter;
}

}  // namespace fpinn
