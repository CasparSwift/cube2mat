#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_range_up_down_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                  const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_high") || !cubes_map.contains("interval_low") || !cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'interval_high', 'interval_low', 'last_price'");
    }
    auto high_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr   = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    auto pbuf = price_arr.request();
    if (hbuf.ndim != 3 || lbuf.ndim != 3 || pbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D arrays");
    }
    if (hbuf.shape[0] != lbuf.shape[0] || hbuf.shape[1] != lbuf.shape[1] || hbuf.shape[2] != lbuf.shape[2] ||
        hbuf.shape[0] != pbuf.shape[0] || hbuf.shape[1] != pbuf.shape[1] || hbuf.shape[2] != pbuf.shape[2]) {
        throw std::runtime_error("arrays must have same shape");
    }
    const ssize_t d0 = hbuf.shape[0];
    const ssize_t d1 = hbuf.shape[1];
    const ssize_t d2 = hbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* high_ptr  = static_cast<float*>(hbuf.ptr);
    const float* low_ptr   = static_cast<float*>(lbuf.ptr);
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double up = 0.0;
            double dn = 0.0;
            for (ssize_t t = 1; t < d2; ++t) {
                float open  = price_ptr[base + t - 1];
                float close = price_ptr[base + t];
                float high  = high_ptr[base + t];
                float low   = low_ptr[base + t];
                if (std::isnan(open) || std::isnan(close) || std::isnan(high) || std::isnan(low) || !(high > low)) continue;
                float range = high - low;
                if (close > open) up += range;
                else if (close < open) dn += range;
            }
            if (!(dn > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            } else {
                res_ptr[i * d1 + j] = static_cast<float>(up / dn);
            }
        }
    }
}

void bind_cube2mat_range_up_down_ratio(py::module& m) {
    m.def("cube2mat_range_up_down_ratio", &cube2mat_range_up_down_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Sum(high-low) on up bars divided by that on down bars.");
}

