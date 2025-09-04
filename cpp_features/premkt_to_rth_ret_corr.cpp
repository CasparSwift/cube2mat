#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_premkt_to_rth_ret_corr(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be a 3D array");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float open_p = price_ptr[base];
            float close_p = price_ptr[base + d2 - 1];
            float prev_close = (i > 0) ? price_ptr[(i - 1) * (d1 * d2) + j * d2 + d2 - 1]
                                      : std::numeric_limits<float>::quiet_NaN();
            if (i == 0 || !(open_p > 0.f) || !(close_p > 0.f) || !(prev_close > 0.f) ||
                std::isnan(open_p) || std::isnan(close_p) || std::isnan(prev_close)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double pre = std::log(open_p) - std::log(prev_close);
            double rth = std::log(close_p) - std::log(open_p);
            int sp = (pre > 0) - (pre < 0);
            int sr = (rth > 0) - (rth < 0);
            res_ptr[i * d1 + j] = static_cast<float>(sp * sr);
        }
    }
}

void bind_cube2mat_premkt_to_rth_ret_corr(py::module& m) {
    m.def("cube2mat_premkt_to_rth_ret_corr", &cube2mat_premkt_to_rth_ret_corr,
          py::arg("result"), py::arg("cubes_map"),
          "Sign consistency proxy between pre-market gap and RTH log returns.");
}

