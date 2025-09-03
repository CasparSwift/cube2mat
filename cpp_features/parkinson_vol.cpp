#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_parkinson_vol(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                            const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_high") || !cubes_map.contains("interval_low")) {
        throw std::runtime_error("cubes_map must contain 'interval_high' and 'interval_low'");
    }
    auto high_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    if (hbuf.ndim != 3 || lbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D arrays");
    }
    const ssize_t d0 = hbuf.shape[0];
    const ssize_t d1 = hbuf.shape[1];
    const ssize_t d2 = hbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* high_ptr = static_cast<float*>(hbuf.ptr);
    const float* low_ptr  = static_cast<float*>(lbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float high = high_ptr[base];
            float low  = low_ptr[base];
            for (ssize_t t = 1; t < d2; ++t) {
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                if (std::isfinite(h) && h > high) high = h;
                if (std::isfinite(l) && l < low)  low  = l;
            }
            if (std::isfinite(high) && std::isfinite(low) && high > 0.0f && low > 0.0f) {
                double loghl = std::log(static_cast<double>(high / low));
                double vol = std::sqrt((loghl * loghl) / (4.0 * std::log(2.0)));
                res_ptr[i * d1 + j] = static_cast<float>(vol);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_parkinson_vol(py::module& m) {
    m.def("cube2mat_parkinson_vol", &cube2mat_parkinson_vol,
          py::arg("result"), py::arg("cubes_map"),
          "Parkinson volatility estimator using session high/low.");
}
