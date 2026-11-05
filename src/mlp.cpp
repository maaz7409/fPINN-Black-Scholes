#include "fpinn/mlp.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace fpinn {

MLP::MLP(std::vector<int> layer_sizes) : sizes_(std::move(layer_sizes)) {
    if (sizes_.size() < 2) throw std::invalid_argument("MLP: need at least input and output layer");
    for (int s : sizes_) {
        if (s < 1) throw std::invalid_argument("MLP: layer sizes must be positive");
    }
    if (sizes_.back() != 1) throw std::invalid_argument("MLP: output layer must have size 1");
    build_offsets();
}

void MLP::build_offsets() {
    w_off_.clear();
    b_off_.clear();
    std::size_t off = 0;
    for (int l = 0; l < num_layers(); ++l) {
        w_off_.push_back(off);
        off += static_cast<std::size_t>(sizes_[l]) * sizes_[l + 1];
        b_off_.push_back(off);
        off += sizes_[l + 1];
    }
    params_.assign(off, 0.0);
}

void MLP::init_xavier(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int l = 0; l < num_layers(); ++l) {
        const int nin = sizes_[l], nout = sizes_[l + 1];
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (nin + nout)));
        for (int k = 0; k < nin * nout; ++k) params_[w_off_[l] + k] = dist(rng);
        for (int k = 0; k < nout; ++k) params_[b_off_[l] + k] = 0.0;
    }
}

MLP::Workspace MLP::make_workspace() const {
    Workspace ws;
    const int nl = num_layers();
    ws.h.resize(nl + 1);
    ws.hd.resize(nl + 1);
    ws.hdd.resize(nl + 1);
    ws.zd.resize(nl + 1);
    ws.zdd.resize(nl + 1);
    int wmax = 0;
    for (int l = 0; l <= nl; ++l) {
        ws.h[l].assign(sizes_[l], 0.0);
        ws.hd[l].assign(sizes_[l], 0.0);
        ws.hdd[l].assign(sizes_[l], 0.0);
        ws.zd[l].assign(sizes_[l], 0.0);
        ws.zdd[l].assign(sizes_[l], 0.0);
        wmax = std::max(wmax, sizes_[l]);
    }
    for (auto* v : {&ws.g, &ws.gd, &ws.gdd, &ws.ga, &ws.gad, &ws.gadd}) v->assign(wmax, 0.0);
    return ws;
}

// Forward pass.  For every layer l:
//   z   = W a + b          z'  = W a'          z''  = W a''
//   h   = tanh(z)          h'  = t1 z'         h''  = t2 z'^2 + t1 z''
// with t1 = tanh'(z) = 1 - h^2 and t2 = tanh''(z) = -2 h t1.
// The last layer is linear (h = z).
Jet MLP::forward(const double* in, const double* dir, Workspace& ws) const {
    const int nl = num_layers();
    for (int k = 0; k < sizes_[0]; ++k) {
        ws.h[0][k] = in[k];
        ws.hd[0][k] = dir[k];
        ws.hdd[0][k] = 0.0;  // the input map is affine
    }
    for (int l = 0; l < nl; ++l) {
        const int nin = sizes_[l], nout = sizes_[l + 1];
        const double* W = params_.data() + w_off_[l];
        const double* b = params_.data() + b_off_[l];
        const double* a = ws.h[l].data();
        const double* ad = ws.hd[l].data();
        const double* add = ws.hdd[l].data();
        const bool last = (l == nl - 1);
        for (int o = 0; o < nout; ++o) {
            const double* row = W + static_cast<std::size_t>(o) * nin;
            double z = b[o], zd = 0.0, zdd = 0.0;
            for (int k = 0; k < nin; ++k) {
                z += row[k] * a[k];
                zd += row[k] * ad[k];
                zdd += row[k] * add[k];
            }
            if (last) {
                ws.h[l + 1][o] = z;
                ws.hd[l + 1][o] = zd;
                ws.hdd[l + 1][o] = zdd;
            } else {
                const double t = std::tanh(z);
                const double t1 = 1.0 - t * t;
                const double t2 = -2.0 * t * t1;
                ws.h[l + 1][o] = t;
                ws.hd[l + 1][o] = t1 * zd;
                ws.hdd[l + 1][o] = t2 * zd * zd + t1 * zdd;
                ws.zd[l + 1][o] = zd;
                ws.zdd[l + 1][o] = zdd;
            }
        }
    }
    return {ws.h[nl][0], ws.hd[nl][0], ws.hdd[nl][0]};
}

double MLP::eval(const double* in, Workspace& ws) const {
    const int nl = num_layers();
    for (int k = 0; k < sizes_[0]; ++k) ws.h[0][k] = in[k];
    for (int l = 0; l < nl; ++l) {
        const int nin = sizes_[l], nout = sizes_[l + 1];
        const double* W = params_.data() + w_off_[l];
        const double* b = params_.data() + b_off_[l];
        const double* a = ws.h[l].data();
        for (int o = 0; o < nout; ++o) {
            const double* row = W + static_cast<std::size_t>(o) * nin;
            double z = b[o];
            for (int k = 0; k < nin; ++k) z += row[k] * a[k];
            ws.h[l + 1][o] = (l == nl - 1) ? z : std::tanh(z);
        }
    }
    return ws.h[nl][0];
}

// Reverse pass through the forward computation above.
// (g, gd, gdd) hold the adjoints of (z, z', z'') of the current layer.
// Linear part:   dW += g a^T + gd a'^T + gdd a''^T,  db += g,
//                adjoints of (a, a', a'') are W^T g, W^T gd, W^T gdd.
// tanh part (h = tanh z), with t3 = tanh'''(z) = -2 t1^2 - 2 h t2:
//   g_z   = g_h t1 + g_h' t2 z' + g_h'' (t3 z'^2 + t2 z'')
//   g_z'  = g_h' t1 + 2 g_h'' t2 z'
//   g_z'' = g_h'' t1
void MLP::backward(const Jet& seed, Workspace& ws, double* grad) const {
    const int nl = num_layers();
    double* g = ws.g.data();
    double* gd = ws.gd.data();
    double* gdd = ws.gdd.data();
    double* ga = ws.ga.data();
    double* gad = ws.gad.data();
    double* gadd = ws.gadd.data();
    g[0] = seed.v;
    gd[0] = seed.d;
    gdd[0] = seed.dd;

    for (int l = nl - 1; l >= 0; --l) {
        const int nin = sizes_[l], nout = sizes_[l + 1];
        const double* W = params_.data() + w_off_[l];
        double* gW = grad + w_off_[l];
        double* gb = grad + b_off_[l];
        const double* a = ws.h[l].data();
        const double* ad = ws.hd[l].data();
        const double* add = ws.hdd[l].data();

        for (int o = 0; o < nout; ++o) {
            gb[o] += g[o];
            double* grow = gW + static_cast<std::size_t>(o) * nin;
            const double go = g[o], gdo = gd[o], gddo = gdd[o];
            for (int k = 0; k < nin; ++k) grow[k] += go * a[k] + gdo * ad[k] + gddo * add[k];
        }
        if (l == 0) break;

        for (int k = 0; k < nin; ++k) {
            ga[k] = 0.0;
            gad[k] = 0.0;
            gadd[k] = 0.0;
        }
        for (int o = 0; o < nout; ++o) {
            const double* row = W + static_cast<std::size_t>(o) * nin;
            const double go = g[o], gdo = gd[o], gddo = gdd[o];
            for (int k = 0; k < nin; ++k) {
                ga[k] += row[k] * go;
                gad[k] += row[k] * gdo;
                gadd[k] += row[k] * gddo;
            }
        }
        // Through the tanh of layer l.
        for (int k = 0; k < nin; ++k) {
            const double t = a[k];
            const double t1 = 1.0 - t * t;
            const double t2 = -2.0 * t * t1;
            const double t3 = -2.0 * t1 * t1 - 2.0 * t * t2;
            const double zd = ws.zd[l][k];
            const double zdd = ws.zdd[l][k];
            g[k] = ga[k] * t1 + gad[k] * t2 * zd + gadd[k] * (t3 * zd * zd + t2 * zdd);
            gd[k] = gad[k] * t1 + 2.0 * gadd[k] * t2 * zd;
            gdd[k] = gadd[k] * t1;
        }
    }
}

void MLP::save(std::ostream& os) const {
    os << "fpinn-mlp 1\n" << sizes_.size();
    for (int s : sizes_) os << ' ' << s;
    os << '\n' << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (double p : params_) os << p << '\n';
}

MLP MLP::load(std::istream& is) {
    std::string magic;
    int version = 0;
    std::size_t n = 0;
    if (!(is >> magic >> version >> n) || magic != "fpinn-mlp" || version != 1) {
        throw std::runtime_error("MLP::load: not an fpinn-mlp file");
    }
    std::vector<int> sizes(n);
    for (auto& s : sizes) is >> s;
    MLP net(sizes);
    for (auto& p : net.params_) {
        if (!(is >> p)) throw std::runtime_error("MLP::load: truncated parameter list");
    }
    return net;
}

}  // namespace fpinn
