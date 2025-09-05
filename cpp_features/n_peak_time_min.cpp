#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_n_peak_time_min(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                              const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_n")) {
        throw std::runtime_error("cubes_map must contain 'interval_n'");
    }
    auto n_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_n"]);
    auto nbuf = n_arr.request();
    if (nbuf.ndim != 3) {
        throw std::runtime_error("interval_n must be 3D");
    }
    const ssize_t d0 = nbuf.shape[0];
    const ssize_t d1 = nbuf.shape[1];
    const ssize_t d2 = nbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* n_ptr = static_cast<float*>(nbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double maxv = -std::numeric_limits<double>::infinity();
            ssize_t idx = -1;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = n_ptr[base + t];
                if (std::isnan(v)) continue;
                double dv = static_cast<double>(v);
                if (dv > maxv) { maxv = dv; idx = t; }
            }
            if (idx >= 0 && maxv > 0.0 && d2 > 1) {
                double frac = static_cast<double>(idx) / static_cast<double>(d2 - 1);
                if (frac < 0.0) frac = 0.0; if (frac > 1.0) frac = 1.0;
                res_ptr[i * d1 + j] = static_cast<float>(frac);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_n_peak_time_min(py::module& m) {
    m.def("cube2mat_n_peak_time_min", &cube2mat_n_peak_time_min,
          py::arg("result"), py::arg("cubes_map"),
          "Fraction of session elapsed (by minutes) when per-bar n peaks within 09:30–15:59.");
}

