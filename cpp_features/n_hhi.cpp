#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_n_hhi(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            double total = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = n_ptr[base + t];
                if (std::isnan(v)) continue;
                total += static_cast<double>(v);
            }
            if (!(total > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double hhi = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = n_ptr[base + t];
                if (std::isnan(v)) continue;
                double p = static_cast<double>(v) / total;
                hhi += p * p;
            }
            res_ptr[i * d1 + j] = static_cast<float>(hhi);
        }
    }
}

void bind_cube2mat_n_hhi(py::module& m) {
    m.def("cube2mat_n_hhi", &cube2mat_n_hhi,
          py::arg("result"), py::arg("cubes_map"),
          "Herfindahl index of trade count (n) distribution across intraday bars.");
}

