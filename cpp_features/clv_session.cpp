#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_clv_session(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                          const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_high") || !cubes_map.contains("interval_low") || !cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'interval_high', 'interval_low', and 'last_price'");
    }
    auto high_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto close_arr= py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto hbuf = high_arr.request();
    auto lbuf = low_arr.request();
    auto cbuf = close_arr.request();
    if (hbuf.ndim!=3 || lbuf.ndim!=3 || cbuf.ndim!=3 ||
        hbuf.shape[0]!=lbuf.shape[0] || hbuf.shape[1]!=lbuf.shape[1] || hbuf.shape[2]!=lbuf.shape[2] ||
        cbuf.shape[0]!=hbuf.shape[0] || cbuf.shape[1]!=hbuf.shape[1] || cbuf.shape[2]!=hbuf.shape[2]) {
        throw std::runtime_error("arrays must be 3D with same shape");
    }
    const ssize_t d0 = hbuf.shape[0];
    const ssize_t d1 = hbuf.shape[1];
    const ssize_t d2 = hbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* high_ptr = static_cast<float*>(hbuf.ptr);
    const float* low_ptr  = static_cast<float*>(lbuf.ptr);
    const float* close_ptr= static_cast<float*>(cbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double H = -std::numeric_limits<double>::infinity();
            double L = std::numeric_limits<double>::infinity();
            for (ssize_t t = 0; t < d2; ++t) {
                float h = high_ptr[base + t];
                float l = low_ptr[base + t];
                if (!std::isnan(h) && h > H) H = h;
                if (!std::isnan(l) && l < L) L = l;
            }
            float c = close_ptr[base + d2 - 1];
            if (!std::isfinite(H) || !std::isfinite(L) || !(H > L) || std::isnan(c)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            res_ptr[i * d1 + j] = static_cast<float>(((c - L) - (H - c)) / (H - L));
        }
    }
}

void bind_cube2mat_clv_session(py::module& m) {
    m.def("cube2mat_clv_session", &cube2mat_clv_session,
          py::arg("result"), py::arg("cubes_map"),
          "Session Close Location Value in RTH using session H/L and last close.");
}

