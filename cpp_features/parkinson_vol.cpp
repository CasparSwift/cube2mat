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
    if (hbuf.shape[0] != lbuf.shape[0] || hbuf.shape[1] != lbuf.shape[1] || hbuf.shape[2] != lbuf.shape[2]) {
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
            double lnHL = std::log(static_cast<double>(H) / static_cast<double>(L));
            double sigma2 = (lnHL * lnHL) / (4.0 * std::log(2.0));
            res_ptr[i * d1 + j] = static_cast<float>(std::sqrt(sigma2));
        }
    }
}

void bind_cube2mat_parkinson_vol(py::module& m) {
    m.def("cube2mat_parkinson_vol", &cube2mat_parkinson_vol,
          py::arg("result"), py::arg("cubes_map"),
          "Parkinson volatility from session high/low: sqrt((ln(H/L))^2 / (4 ln 2)).");
}

