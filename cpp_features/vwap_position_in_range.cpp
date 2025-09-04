#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_vwap_position_in_range(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_high") || !cubes_map.contains("interval_low") ||
        !cubes_map.contains("interval_vwap") || !cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_high', 'interval_low', 'interval_vwap', 'interval_volume'");
    }
    auto high_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto vwap_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto vol_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    auto vwbuf = vwap_arr.request();
    auto vbuf = vol_arr.request();
    if (hbuf.ndim != 3 || lbuf.ndim != 3 || vwbuf.ndim != 3 || vbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D arrays");
    }
    if (hbuf.shape[0] != lbuf.shape[0] || hbuf.shape[1] != lbuf.shape[1] || hbuf.shape[2] != lbuf.shape[2] ||
        hbuf.shape[0] != vwbuf.shape[0] || hbuf.shape[1] != vwbuf.shape[1] || hbuf.shape[2] != vwbuf.shape[2] ||
        hbuf.shape[0] != vbuf.shape[0] || hbuf.shape[1] != vbuf.shape[1] || hbuf.shape[2] != vbuf.shape[2]) {
        throw std::runtime_error("arrays must have same shape");
    }
    const ssize_t d0 = hbuf.shape[0];
    const ssize_t d1 = hbuf.shape[1];
    const ssize_t d2 = hbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* high_ptr = static_cast<float*>(hbuf.ptr);
    const float* low_ptr  = static_cast<float*>(lbuf.ptr);
    const float* vwap_ptr = static_cast<float*>(vwbuf.ptr);
    const float* vol_ptr  = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float H = -std::numeric_limits<float>::infinity();
            float L =  std::numeric_limits<float>::infinity();
            double num = 0.0;
            double den = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                float vw = vwap_ptr[base + t];
                float v = vol_ptr[base + t];
                if (!std::isnan(h) && h > H) H = h;
                if (!std::isnan(l) && l < L) L = l;
                if (!std::isnan(vw) && !std::isnan(v) && v > 0.0f) {
                    num += vw * v;
                    den += v;
                }
            }
            float rng = H - L;
            if (!(rng > 0.0f) || !(den > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            } else {
                float vw_sess = static_cast<float>(num / den);
                res_ptr[i * d1 + j] = (vw_sess - L) / rng;
            }
        }
    }
}

void bind_cube2mat_vwap_position_in_range(py::module& m) {
    m.def("cube2mat_vwap_position_in_range", &cube2mat_vwap_position_in_range,
          py::arg("result"), py::arg("cubes_map"),
          "Position of session VWAP in the day's H-L range: (VWAP - L)/(H - L).");
}

