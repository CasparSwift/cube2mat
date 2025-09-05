#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_transition_asymmetry(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            int up_start = 0, down_start = 0;
            int upup = 0, downdown = 0;
            for (ssize_t t = 1; t < d2 - 1; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                float p2 = price_ptr[base + t + 1];
                if (!(p0 > 0.f) || !(p1 > 0.f) || !(p2 > 0.f) ||
                    std::isnan(p0) || std::isnan(p1) || std::isnan(p2)) continue;
                float r1 = (p1 - p0) / p0;
                float r2 = (p2 - p1) / p1;
                int s1 = (r1 > 0.f) ? 1 : (r1 < 0.f ? -1 : 0);
                int s2 = (r2 > 0.f) ? 1 : (r2 < 0.f ? -1 : 0);
                if (s1 == 0 || s2 == 0) continue;
                if (s1 > 0) { ++up_start; if (s2 > 0) ++upup; }
                else if (s1 < 0) { ++down_start; if (s2 < 0) ++downdown; }
            }
            if (up_start == 0 || down_start == 0) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
            } else {
                float p_up = static_cast<float>(upup) / up_start;
                float p_down = static_cast<float>(downdown) / down_start;
                res_ptr[i*d1 + j] = p_up - p_down;
            }
        }
    }
}

void bind_cube2mat_transition_asymmetry(py::module& m) {
    m.def("cube2mat_transition_asymmetry", &cube2mat_transition_asymmetry,
          py::arg("result"), py::arg("cubes_map"),
          "P(up→up) - P(down→down) from simple-return signs (exclude zeros).");
}

