#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float gini_coeff(std::vector<float>& vals) {
    const size_t n = vals.size();
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();
    std::sort(vals.begin(), vals.end());
    double sum = 0.0;
    double cumsum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += vals[i];
        cumsum += static_cast<double>(i + 1) * vals[i];
    }
    if (sum <= 0.0) return std::numeric_limits<float>::quiet_NaN();
    double g = (2.0 * cumsum) / (n * sum) - (static_cast<double>(n + 1)) / n;
    return static_cast<float>(g);
}

void cube2mat_trade_size_gini(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                              const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume") || !cubes_map.contains("interval_n")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume' and 'interval_n'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto n_arr   = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_n"]);
    auto vbuf = vol_arr.request();
    auto nbuf = n_arr.request();
    if (vbuf.ndim != 3 || nbuf.ndim != 3) {
        throw std::runtime_error("interval_volume and interval_n must be 3D arrays");
    }
    if (vbuf.shape[0] != nbuf.shape[0] || vbuf.shape[1] != nbuf.shape[1] || vbuf.shape[2] != nbuf.shape[2]) {
        throw std::runtime_error("interval_volume and interval_n must have same shape");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    const float* n_ptr   = static_cast<float*>(nbuf.ptr);
    float* res_ptr       = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals; vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                float n = n_ptr[base + t];
                if (!std::isnan(v) && !std::isnan(n) && v > 0.0f && n > 0.0f) {
                    vals.push_back(v / n);
                }
            }
            res_ptr[i * d1 + j] = gini_coeff(vals);
        }
    }
}

void bind_cube2mat_trade_size_gini(py::module& m) {
    m.def("cube2mat_trade_size_gini", &cube2mat_trade_size_gini,
          py::arg("result"), py::arg("cubes_map"),
          "Gini of per-bar average trade size across RTH bars.");
}

