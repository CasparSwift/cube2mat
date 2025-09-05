#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_time_share_above_vwap(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                   const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    }

    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf = price_arr.request();
    auto vbuf = vwap_arr.request();
    if (pbuf.ndim != 3 || vbuf.ndim != 3) {
        throw std::runtime_error("last_price and interval_vwap must be 3D");
    }
    if (pbuf.shape[0] != vbuf.shape[0] || pbuf.shape[1] != vbuf.shape[1] || pbuf.shape[2] != vbuf.shape[2]) {
        throw std::runtime_error("last_price and interval_vwap must have same shape");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }

    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr  = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            int cnt = 0;
            int tot = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                float v = vwap_ptr[base + t];
                if (std::isnan(c) || std::isnan(v)) continue;
                ++tot;
                if (c > v) ++cnt;
            }
            if (tot > 0) {
                res_ptr[i * d1 + j] = static_cast<float>(static_cast<double>(cnt) / tot);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_time_share_above_vwap(py::module& m) {
    m.def("cube2mat_time_share_above_vwap", &cube2mat_time_share_above_vwap,
          py::arg("result"), py::arg("cubes_map"),
          "Fraction of bars with close > vwap during the session.");
}

