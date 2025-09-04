#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_cond_reversal_after_run2(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
        throw std::runtime_error("result must be 2D (d0,d1) matching last_price leading dims");
    }
    const float* close_ptr = static_cast<float*>(cbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            // compute simple returns
            std::vector<float> sgn(d2 - 1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = (p1 - p0) / p0;
                if (!std::isfinite(r)) r = std::numeric_limits<float>::quiet_NaN();
                sgn[t - 1] = (r > 0.0f) ? 1.0f : (r < 0.0f ? -1.0f : 0.0f);
            }
            ssize_t n = d2 - 1;
            ssize_t events = 0;
            ssize_t reversals = 0;
            for (ssize_t t = 1; t < n - 1; ++t) {
                float a = sgn[t - 1];
                float b = sgn[t];
                float c = sgn[t + 1];
                if (a == b && b != 0.0f && c != 0.0f) {
                    ++events;
                    if (c != b) ++reversals;
                }
            }
            if (events < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            } else {
                res_ptr[i * d1 + j] = static_cast<float>(static_cast<double>(reversals) / events);
            }
        }
    }
}

void bind_cube2mat_cond_reversal_after_run2(py::module& m) {
    m.def("cube2mat_cond_reversal_after_run2", &cube2mat_cond_reversal_after_run2,
          py::arg("result"), py::arg("cubes_map"),
          "P(reversal | two consecutive same-sign simple returns).");
}

