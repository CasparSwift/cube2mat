#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_n_peak_time_frac(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    const ssize_t span = d2 - 1;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float maxv = -std::numeric_limits<float>::infinity();
            ssize_t argmax = -1;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isnan(v) || !(v > maxv)) continue;
                maxv = v;
                argmax = t;
            }
            if (argmax >= 0 && maxv > 0.0f && span > 0) {
                res_ptr[i * d1 + j] = static_cast<float>(argmax / double(span));
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_n_peak_time_frac(py::module& m) {
    m.def("cube2mat_n_peak_time_frac", &cube2mat_n_peak_time_frac,
          py::arg("result"), py::arg("cubes_map"),
          "Fractional bar-position when per-bar n peaks within 09:30–15:59.");
}

