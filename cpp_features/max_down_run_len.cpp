#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_max_down_run_len(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto close_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto cbuf = close_arr.request();
    if (cbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* close_ptr = static_cast<float*>(cbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            ssize_t max_run = 0;
            ssize_t cur = 0;
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = (p1 - p0) / p0;
                if (r < 0.0f && std::isfinite(r)) {
                    ++cur;
                    if (cur > max_run) max_run = cur;
                } else {
                    cur = 0;
                }
            }
            res_ptr[i * d1 + j] = (max_run > 0) ? static_cast<float>(max_run)
                                                 : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_max_down_run_len(py::module& m) {
    m.def("cube2mat_max_down_run_len", &cube2mat_max_down_run_len,
          py::arg("result"), py::arg("cubes_map"),
          "Maximum consecutive length of negative-return run.");
}

