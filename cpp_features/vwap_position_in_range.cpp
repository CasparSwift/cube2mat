#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_vwap_position_in_range(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_vwap") || !cubes_map.contains("interval_volume") || !cubes_map.contains("interval_high") || !cubes_map.contains("interval_low")) {
        throw std::runtime_error("cubes_map must contain 'interval_vwap','interval_volume','interval_high','interval_low'");
    }
    auto vwap_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto vol_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto high_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto vwbuf = vwap_arr.request();
    auto vbuf = vol_arr.request();
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    if (vwbuf.ndim != 3 || vbuf.ndim != 3 || hbuf.ndim != 3 || lbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D arrays");
    }
    const ssize_t d0 = vwbuf.shape[0];
    const ssize_t d1 = vwbuf.shape[1];
    const ssize_t d2 = vwbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* vwap_ptr = static_cast<float*>(vwbuf.ptr);
    const float* vol_ptr  = static_cast<float*>(vbuf.ptr);
    const float* high_ptr = static_cast<float*>(hbuf.ptr);
    const float* low_ptr  = static_cast<float*>(lbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double num = 0.0, den = 0.0;
            float high = high_ptr[base];
            float low  = low_ptr[base];
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                float w = vwap_ptr[base + t];
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                if (std::isfinite(v) && v > 0.0f && std::isfinite(w)) {
                    num += w * v;
                    den += v;
                }
                if (std::isfinite(h) && h > high) high = h;
                if (std::isfinite(l) && l < low)  low  = l;
            }
            float vwap = (den > 0.0) ? static_cast<float>(num / den) : std::numeric_limits<float>::quiet_NaN();
            float range = high - low;
            float val = (std::isfinite(vwap) && range > 0.0f) ? (vwap - low) / range
                                                              : std::numeric_limits<float>::quiet_NaN();
            res_ptr[i * d1 + j] = val;
        }
    }
}

void bind_vwap_position_in_range(py::module& m) {
    m.def("cube2mat_vwap_position_in_range", &cube2mat_vwap_position_in_range,
          py::arg("result"), py::arg("cubes_map"),
          "Position of session VWAP within range.");
}
