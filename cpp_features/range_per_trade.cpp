#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_range_per_trade(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                              const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_high") || !cubes_map.contains("interval_low") || !cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_high','interval_low','interval_volume'");
    }
    auto high_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto vol_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    auto vbuf = vol_arr.request();
    if (hbuf.ndim != 3 || lbuf.ndim != 3 || vbuf.ndim != 3) {
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
    const float* vol_ptr  = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float high = high_ptr[base];
            float low  = low_ptr[base];
            double volsum = 0.0;
            for (ssize_t t = 1; t < d2; ++t) {
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                if (std::isfinite(h) && h > high) high = h;
                if (std::isfinite(l) && l < low)  low  = l;
            }
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v > 0.0f) volsum += v;
            }
            float range = high - low;
            res_ptr[i * d1 + j] = (volsum > 0.0 && range > 0.0f) ? static_cast<float>(range / volsum)
                                                                 : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_range_per_trade(py::module& m) {
    m.def("cube2mat_range_per_trade", &cube2mat_range_per_trade,
          py::arg("result"), py::arg("cubes_map"),
          "Session range divided by total trade count (volume proxy).");
}
