#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_breakout_nextret_high(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                    const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
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
            double sum = 0.0;
            int count = 0;
            float prev_high = -std::numeric_limits<float>::infinity();
            for (ssize_t t = 0; t < d2 - 1; ++t) {
                float p = price_ptr[base + t];
                if (!(p > 0.0f) || std::isnan(p)) continue;
                if (p > prev_high) {
                    float pnext = price_ptr[base + t + 1];
                    if (pnext > 0.0f && !std::isnan(pnext)) {
                        sum += (pnext / p) - 1.0f;
                        ++count;
                    }
                    prev_high = p;
                }
            }
            res_ptr[i * d1 + j] = (count > 0)
                ? static_cast<float>(sum / count)
                : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_breakout_nextret_high(py::module& m) {
    m.def("cube2mat_breakout_nextret_high", &cube2mat_breakout_nextret_high,
          py::arg("result"), py::arg("cubes_map"),
          "Mean next simple return when making new session high.");
}

