#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_share_in_runs_len_ge3(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                    const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1)
        throw std::runtime_error("result must be (d0,d1)");

    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i*(d1*d2) + j*d2;
            int denom = 0; int num = 0;
            int cur_sign = 0; int run_len = 0;
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) {
                    if (run_len >= 3) num += run_len;
                    cur_sign = 0; run_len = 0; continue;
                }
                float r = (p1 - p0) / p0;
                int s = (r > 0.f) ? 1 : (r < 0.f ? -1 : 0);
                if (s == 0) {
                    if (run_len >= 3) num += run_len;
                    cur_sign = 0; run_len = 0;
                    continue;
                }
                ++denom;
                if (s == cur_sign) {
                    ++run_len;
                } else {
                    if (run_len >= 3) num += run_len;
                    cur_sign = s; run_len = 1;
                }
            }
            if (run_len >= 3) num += run_len;
            res_ptr[i*d1 + j] = (denom > 0)
                ? (static_cast<float>(num) / denom)
                : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_share_in_runs_len_ge3(py::module& m) {
    m.def("cube2mat_share_in_runs_len_ge3", &cube2mat_share_in_runs_len_ge3,
          py::arg("result"), py::arg("cubes_map"),
          "Share of return observations in sign runs with length ≥ 3.");
}

