#include "fpinn/special_functions.hpp"

#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace fpinn {

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); }

double mittag_leffler(double alpha, double z) {
    if (!(alpha > 0.0 && alpha <= 1.0)) {
        throw std::invalid_argument("mittag_leffler: alpha must be in (0, 1]");
    }
    if (z == 0.0) return 1.0;
    if (alpha == 1.0) return std::exp(z);

    const double absz = std::abs(z);
    // The largest series term is roughly exp(|z|^(1/alpha)).  For negative z the
    // terms alternate, so we lose about that many digits to cancellation.
    const double peak = std::pow(absz, 1.0 / alpha);
    if (peak > 18.0) {
        throw std::domain_error("mittag_leffler: |z| too large for the power series");
    }

    const double log_absz = std::log(absz);
    double sum = 0.0;
    const int min_terms = static_cast<int>(2.0 * peak / alpha) + 10;
    for (int k = 0; k < 5000; ++k) {
        const double term = std::exp(k * log_absz - std::lgamma(alpha * k + 1.0));
        const double sign = (z < 0.0 && (k % 2 == 1)) ? -1.0 : 1.0;
        sum += sign * term;
        if (k > min_terms && term < 1e-17 * std::max(1.0, std::abs(sum))) break;
    }
    return sum;
}

namespace {

// Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations.
// a (n x n, row-major) is destroyed; returns eigenvalues, v receives eigenvectors as columns.
std::vector<double> jacobi_eigen(std::vector<double> a, int n, std::vector<double>& v) {
    v.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) v[i * n + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) off += a[i * n + j] * a[i * n + j];
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                if (std::abs(apq) < 1e-300) continue;
                const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {  // A <- A J
                    const double akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {  // A <- J^T A
                    const double apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {  // V <- V J
                    const double vkp = v[k * n + p], vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<double> ev(n);
    for (int i = 0; i < n; ++i) ev[i] = a[i * n + i];
    return ev;
}

}  // namespace

void random_clock_gauss_rule(double alpha, int n, std::vector<double>& nodes, std::vector<double>& weights) {
    if (!(alpha > 0.0 && alpha <= 1.0)) throw std::invalid_argument("random_clock_gauss_rule: alpha in (0, 1]");
    if (n < 1) throw std::invalid_argument("random_clock_gauss_rule: n >= 1");
    if (alpha == 1.0) {
        nodes.assign(1, 1.0);
        weights.assign(1, 1.0);
        return;
    }
    // Moments mu_k = k! / Gamma(1 + alpha k), k = 0 .. 2n-1.
    const int m = 2 * n;
    std::vector<long double> mu(m);
    for (int k = 0; k < m; ++k) mu[k] = std::exp(std::lgamma(k + 1.0L) - std::lgamma(1.0L + alpha * k));

    // Gautschi's Chebyshev algorithm: moments -> three-term recurrence (a_k, b_k).
    std::vector<long double> a(n), b(n);
    std::vector<long double> sig_prev(m, 0.0L), sig(mu.begin(), mu.end()), sig_next(m, 0.0L);
    a[0] = mu[1] / mu[0];
    b[0] = mu[0];
    int used = 1;
    for (int k = 1; k < n; ++k) {
        for (int l = k; l <= m - k - 1; ++l) {
            sig_next[l] = sig[l + 1] - a[k - 1] * sig[l] - b[k - 1] * sig_prev[l];
        }
        const long double bk = sig_next[k] / sig[k - 1];
        // Stop if the recurrence loses positivity / accuracy (nearly degenerate law, alpha ~ 1).
        if (!(bk > 1e-14L * a[0] * a[0])) break;
        b[k] = bk;
        a[k] = sig_next[k + 1] / sig_next[k] - sig[k] / sig[k - 1];
        sig_prev.swap(sig);
        sig.swap(sig_next);
        used = k + 1;
    }

    // Golub-Welsch: nodes = eigenvalues of the Jacobi matrix, weights from eigenvectors.
    std::vector<double> J(static_cast<std::size_t>(used) * used, 0.0), V;
    for (int i = 0; i < used; ++i) {
        J[i * used + i] = static_cast<double>(a[i]);
        if (i + 1 < used) {
            const double off = std::sqrt(static_cast<double>(b[i + 1]));
            J[i * used + i + 1] = off;
            J[(i + 1) * used + i] = off;
        }
    }
    const std::vector<double> ev = jacobi_eigen(J, used, V);
    std::vector<std::pair<double, double>> nw(used);
    for (int q = 0; q < used; ++q) nw[q] = {ev[q], static_cast<double>(mu[0]) * V[q] * V[q]};  // V[0*used+q]
    std::sort(nw.begin(), nw.end());
    nodes.resize(used);
    weights.resize(used);
    for (int q = 0; q < used; ++q) {
        nodes[q] = nw[q].first;
        weights[q] = nw[q].second;
    }
}

}  // namespace fpinn
