#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_next_ret_after_top_absret_decile(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> logrets;
            logrets.reserve(d2-1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                logrets.push_back(std::log(p1 / p0));
            }
            ssize_t n = logrets.size();
            if (n < 2) { res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> abs_lr(n);
            for (ssize_t t = 0; t < n; ++t) abs_lr[t] = std::fabs(logrets[t]);
            std::vector<float> tmp = abs_lr;
            std::sort(tmp.begin(), tmp.end());
            float thresh = tmp[static_cast<size_t>(0.9f*(n-1))];
            float sum = 0.f; int cnt = 0;
            for (ssize_t t = 0; t < n-1; ++t) {
                if (abs_lr[t] >= thresh) {
                    float p_curr = price_ptr[base + t + 1];
                    float p_next = price_ptr[base + t + 2];
                    if (!(p_curr > 0.f) || !(p_next > 0.f) || std::isnan(p_curr) || std::isnan(p_next)) continue;
                    float r_next = (p_next - p_curr) / p_curr;
                    sum += r_next; ++cnt;
                }
            }
            res_ptr[i*d1 + j] = (cnt > 0) ? (sum / cnt) : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_next_ret_after_top_absret_decile(py::module& m) {
    m.def("cube2mat_next_ret_after_top_absret_decile", &cube2mat_next_ret_after_top_absret_decile,
          py::arg("result"), py::arg("cubes_map"),
          "E[ret_{t+1} | |logret_t| in top decile].");
}

