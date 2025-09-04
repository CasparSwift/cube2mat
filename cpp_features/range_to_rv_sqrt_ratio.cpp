#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_range_to_rv_sqrt_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            float H = -std::numeric_limits<float>::infinity();
            float L =  std::numeric_limits<float>::infinity();
            for (ssize_t t = 0; t < d2; ++t) {
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                if (std::isnan(h) || std::isnan(l)) continue;
                if (h > H) H = h;
                if (l < L) L = l;
            }
            if (!(H > 0.0f) || !(L > 0.0f) || !(H > L)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double log_range = std::log(static_cast<double>(H) / static_cast<double>(L));
            double sumsq = 0.0;
            for (ssize_t t = 1; t < d2; ++t) {
                float c0 = price_ptr[base + t - 1];
                float c1 = price_ptr[base + t];
                if (std::isnan(c0) || std::isnan(c1) || c0 <= 0.0f || c1 <= 0.0f) continue;
                double r = std::log(static_cast<double>(c1) / static_cast<double>(c0));
                sumsq += r * r;
            }
            if (!(sumsq > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            } else {
                res_ptr[i * d1 + j] = static_cast<float>(log_range / std::sqrt(sumsq));
            }
        }
    }
}

void bind_cube2mat_range_to_rv_sqrt_ratio(py::module& m) {
    m.def("cube2mat_range_to_rv_sqrt_ratio", &cube2mat_range_to_rv_sqrt_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Log session range log(high_max/low_min) divided by sqrt(Σ r^2) using RTH log-returns.");
}

