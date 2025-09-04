#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float quantile25(std::vector<float>& v) {
    if (v.empty()) return std::numeric_limits<float>::quiet_NaN();
    std::sort(v.begin(), v.end());
    double pos = 0.25 * (v.size() - 1);
    size_t idx = static_cast<size_t>(std::floor(pos));
    double frac = pos - idx;
    float v1 = v[idx];
    float v2 = v[std::min(idx + 1, v.size() - 1)];
    return static_cast<float>(v1 + (v2 - v1) * frac);
}

void cube2mat_quietest_stretch_maxlen_q25_absret_frac(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> absr;
            absr.reserve(d2 - 1);
            std::vector<bool> mask(d2 - 1, false);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = std::log(p1) - std::log(p0);
                if (!std::isfinite(r)) continue;
                absr.push_back(std::fabs(r));
            }
            ssize_t n = absr.size();
            if (n == 0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<float> sorted = absr;
            float thr = quantile25(sorted);
            // build mask
            for (ssize_t k = 0; k < n; ++k) mask[k] = absr[k] <= thr;
            // longest run of true
            ssize_t best = 0, cur = 0;
            for (ssize_t k = 0; k < n; ++k) {
                if (mask[k]) {
                    ++cur;
                    if (cur > best) best = cur;
                } else {
                    cur = 0;
                }
            }
            res_ptr[i * d1 + j] = static_cast<float>(best) / n;
        }
    }
}

void bind_cube2mat_quietest_stretch_maxlen_q25_absret_frac(py::module& m) {
    m.def("cube2mat_quietest_stretch_maxlen_q25_absret_frac", &cube2mat_quietest_stretch_maxlen_q25_absret_frac,
          py::arg("result"), py::arg("cubes_map"),
          "Longest run length fraction where |r| ≤ Q25(|r|) within RTH (log returns).");
}

