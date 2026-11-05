#pragma once

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace fpinn {

/// Value of a scalar function together with its first and second derivative
/// along one fixed input direction (here: d/dx and d2/dx2).
struct Jet {
    double v = 0.0;
    double d = 0.0;
    double dd = 0.0;
};

/// Fully-connected network  R^n -> R  with tanh hidden layers and a linear output.
///
/// Besides the plain output N(z), `forward` returns the first and second
/// directional derivatives  dN/ds, d2N/ds2  along a direction `dir` in input
/// space ("forward-mode / Taylor-mode automatic differentiation").  `backward`
/// then back-propagates through that whole computation, giving the gradient
/// of any loss of the form  g0 * N + g1 * N' + g2 * N''  with respect to every
/// weight.  This is exactly what a physics-informed loss with a second-order
/// PDE needs, and it is written by hand so the project has no ML dependency.
class MLP {
public:
    MLP() = default;
    explicit MLP(std::vector<int> layer_sizes);  ///< e.g. {2, 32, 32, 32, 1}

    const std::vector<int>& layer_sizes() const { return sizes_; }
    int num_layers() const { return static_cast<int>(sizes_.size()) - 1; }
    std::size_t num_params() const { return params_.size(); }
    std::vector<double>& params() { return params_; }
    const std::vector<double>& params() const { return params_; }

    /// Glorot/Xavier normal initialisation of weights, zero biases.
    void init_xavier(std::uint64_t seed);

    /// Scratch memory for one evaluation.  One per thread.
    struct Workspace {
        std::vector<std::vector<double>> h, hd, hdd;  // activations and their derivatives
        std::vector<std::vector<double>> zd, zdd;     // pre-activation derivatives
        std::vector<double> g, gd, gdd, ga, gad, gadd;
    };
    Workspace make_workspace() const;

    /// Evaluate at input `in` with derivative direction `dir` (both of length
    /// layer_sizes()[0]).  Stores intermediates in `ws` for a following backward().
    Jet forward(const double* in, const double* dir, Workspace& ws) const;

    /// Plain evaluation (no derivatives).
    double eval(const double* in, Workspace& ws) const;

    /// Accumulate into `grad` (size num_params()) the gradient of
    ///     seed.v * N + seed.d * N' + seed.dd * N''
    /// for the input most recently passed to forward() with this workspace.
    void backward(const Jet& seed, Workspace& ws, double* grad) const;

    void save(std::ostream& os) const;
    static MLP load(std::istream& is);

private:
    void build_offsets();

    std::vector<int> sizes_;
    std::vector<double> params_;
    std::vector<std::size_t> w_off_, b_off_;
};

}  // namespace fpinn
