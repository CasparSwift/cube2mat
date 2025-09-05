#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_vwap_cross_down_nextret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                      const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf = price_arr.request();
    auto vbuf = vwap_arr.request();
    if (pbuf.ndim != 3 || vbuf.ndim != 3) throw std::runtime_error("arrays must be 3D");
    if (pbuf.shape[0]!=vbuf.shape[0] || pbuf.shape[1]!=vbuf.shape[1] || pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("last_price and interval_vwap must have same shape");
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1)
        throw std::runtime_error("result must be (d0,d1)");

    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr  = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i*(d1*d2) + j*d2;
            float sum = 0.f; int cnt = 0;
            for (ssize_t t = 1; t < d2-1; ++t) {
                float p_prev = price_ptr[base + t - 1];
                float p_curr = price_ptr[base + t];
                float p_next = price_ptr[base + t + 1];
                float v_prev = vwap_ptr[base + t - 1];
                float v_curr = vwap_ptr[base + t];
                if (std::isnan(p_prev) || std::isnan(p_curr) || std::isnan(p_next) ||
                    std::isnan(v_prev) || std::isnan(v_curr) || !(p_prev>0.f) || !(p_curr>0.f) || !(p_next>0.f))
                    continue;
                if ((p_prev - v_prev) >= 0.f && (p_curr - v_curr) < 0.f) {
                    float r_next = (p_next - p_curr) / p_curr;
                    sum += r_next; ++cnt;
                }
            }
            res_ptr[i*d1 + j] = (cnt > 0) ? (sum / cnt) : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_vwap_cross_down_nextret(py::module& m) {
    m.def("cube2mat_vwap_cross_down_nextret", &cube2mat_vwap_cross_down_nextret,
          py::arg("result"), py::arg("cubes_map"),
          "Mean next return after downward VWAP crossing.");
}

