#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_max_drawdown_depth_per_bar(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            bool init = false;
            float peak = 0.0f;
            ssize_t peak_idx = 0;
            bool has_dd = false;
            float best_ratio = 0.0f;
            for (ssize_t t = 0; t < d2; ++t) {
                float p = price_ptr[base + t];
                if (!(p > 0.0f) || std::isnan(p)) continue;
                if (!init || p > peak) {
                    peak = p;
                    peak_idx = t;
                    init = true;
                    continue;
                }
                float depth = peak - p;
                ssize_t duration = t - peak_idx;
                if (duration > 0 && depth > 0.0f) {
                    float ratio = depth / duration;
                    if (!has_dd || ratio > best_ratio) {
                        best_ratio = ratio;
                        has_dd = true;
                    }
                }
            }
            res_ptr[i * d1 + j] = has_dd ? best_ratio : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_max_drawdown_depth_per_bar(py::module& m) {
    m.def("cube2mat_max_drawdown_depth_per_bar", &cube2mat_max_drawdown_depth_per_bar,
          py::arg("result"), py::arg("cubes_map"),
          "Maximum drawdown depth divided by its duration in bars.");
}

