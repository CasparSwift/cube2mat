#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_midrange_cross_count(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                   const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_high") || !cubes_map.contains("interval_low")) {
        throw std::runtime_error("cubes_map must contain 'last_price', 'interval_high', 'interval_low'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto high_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr   = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto pbuf = price_arr.request();
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    if (pbuf.ndim != 3 || hbuf.ndim != 3 || lbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D");
    }
    if (pbuf.shape[0] != hbuf.shape[0] || pbuf.shape[1] != hbuf.shape[1] || pbuf.shape[2] != hbuf.shape[2] ||
        pbuf.shape[0] != lbuf.shape[0] || pbuf.shape[1] != lbuf.shape[1] || pbuf.shape[2] != lbuf.shape[2]) {
        throw std::runtime_error("arrays must have same shape");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* high_ptr  = static_cast<float*>(hbuf.ptr);
    const float* low_ptr   = static_cast<float*>(lbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double H = -std::numeric_limits<double>::infinity();
            double L =  std::numeric_limits<double>::infinity();
            for (ssize_t t = 0; t < d2; ++t) {
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                if (!std::isnan(h)) H = std::max(H, static_cast<double>(h));
                if (!std::isnan(l)) L = std::min(L, static_cast<double>(l));
            }
            if (!(H > L)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mid = (H + L) / 2.0;
            int prev = 0;
            int flips = 0;
            int nz = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                if (std::isnan(c)) continue;
                double diff = static_cast<double>(c) - mid;
                int s = (diff > 0.0) - (diff < 0.0);
                if (s == 0) continue;
                if (prev != 0 && s != prev) ++flips;
                prev = s;
                ++nz;
            }
            res_ptr[i * d1 + j] = (nz >= 2) ? static_cast<float>(flips)
                                             : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_midrange_cross_count(py::module& m) {
    m.def("cube2mat_midrange_cross_count", &cube2mat_midrange_cross_count,
          py::arg("result"), py::arg("cubes_map"),
          "Count of sign flips of (close − midrange) where midrange = (H+L)/2 using session H/L.");
}

