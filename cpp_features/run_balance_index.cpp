#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static std::vector<int> run_lengths(const std::vector<int>& s, bool positive) {
    std::vector<int> lens;
    int cur = 0;
    for (int v : s) {
        bool cond = positive ? (v > 0) : (v < 0);
        if (cond) {
            ++cur;
        } else if (cur > 0) {
            lens.push_back(cur);
            cur = 0;
        }
    }
    if (cur > 0) lens.push_back(cur);
    return lens;
}

void cube2mat_run_balance_index(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                 const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto close_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto cbuf = close_arr.request();
    if (cbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* close_ptr = static_cast<float*>(cbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<int> sgn(d2 - 1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = std::log(p1) - std::log(p0);
                if (!std::isfinite(r)) r = 0.0f;
                sgn[t - 1] = (r > 0.0f) ? 1 : (r < 0.0f ? -1 : 0);
            }
            auto ups = run_lengths(sgn, true);
            auto dns = run_lengths(sgn, false);
            if (ups.empty() || dns.empty()) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mu_u = 0.0, mu_d = 0.0;
            for (int v : ups) mu_u += v;
            mu_u /= ups.size();
            for (int v : dns) mu_d += v;
            mu_d /= dns.size();
            double denom = mu_u + mu_d;
            res_ptr[i * d1 + j] = (denom > 0.0)
                ? static_cast<float>((mu_u - mu_d) / denom)
                : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_run_balance_index(py::module& m) {
    m.def("cube2mat_run_balance_index", &cube2mat_run_balance_index,
          py::arg("result"), py::arg("cubes_map"),
          "Run-balance index: (mean length of up runs − mean length of down runs) / (sum of the two means).");
}

