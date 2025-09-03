#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_path_length_over_range(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_high") || !cubes_map.contains("interval_low")) {
        throw std::runtime_error("cubes_map must contain 'last_price','interval_high','interval_low'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto high_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr   = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto pbuf = price_arr.request();
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    if (pbuf.ndim != 3 || hbuf.ndim != 3 || lbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D arrays");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* high_ptr  = static_cast<float*>(hbuf.ptr);
    const float* low_ptr   = static_cast<float*>(lbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double path = 0.0;
            float prev = price_ptr[base];
            bool prev_valid = std::isfinite(prev);
            float high = high_ptr[base];
            float low  = low_ptr[base];
            for (ssize_t t = 1; t < d2; ++t) {
                float c = price_ptr[base + t];
                bool cv = std::isfinite(c);
                if (prev_valid && cv) path += std::fabs(static_cast<double>(c - prev));
                prev = c;
                prev_valid = cv;
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                if (std::isfinite(h) && h > high) high = h;
                if (std::isfinite(l) && l < low)  low  = l;
            }
            float range = high - low;
            float val = (range > 0.0f) ? static_cast<float>(path / range)
                                       : std::numeric_limits<float>::quiet_NaN();
            res_ptr[i * d1 + j] = std::isfinite(val) ? val : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_path_length_over_range(py::module& m) {
    m.def("cube2mat_path_length_over_range", &cube2mat_path_length_over_range,
          py::arg("result"), py::arg("cubes_map"),
          "Sum |Δclose| normalized by session range.");
}
