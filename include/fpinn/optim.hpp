#pragma once

#include <functional>
#include <vector>

namespace fpinn {

/// Adam optimiser (Kingma & Ba, 2015).
class Adam {
public:
    explicit Adam(std::size_t n, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
    void step(std::vector<double>& params, const std::vector<double>& grad, double lr);

private:
    double b1_, b2_, eps_;
    long t_ = 0;
    std::vector<double> m_, v_;
};

/// f(x, grad) must return the objective value and write its gradient into grad.
using Objective = std::function<double(const std::vector<double>& x, std::vector<double>& grad)>;

struct LBFGSConfig {
    int max_iter = 500;
    int history = 20;
    double grad_tol = 1e-12;  ///< stop when ||grad||_inf falls below this
};

/// Limited-memory BFGS with an Armijo back-tracking line search.
/// Minimises f starting from x (updated in place).  `on_iter(iter, f)` is called
/// after each accepted step.  Returns the number of iterations performed.
int lbfgs_minimize(const Objective& f, std::vector<double>& x, const LBFGSConfig& cfg,
                   const std::function<void(int, double)>& on_iter = {});

}  // namespace fpinn
