#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_n_entropy_concentration(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                      const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_n")) {
        throw std::runtime_error("cubes_map must contain 'interval_n'");
    }
    auto n_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_n"]);
    auto nbuf = n_arr.request();
    if (nbuf.ndim != 3) {
        throw std::runtime_error("interval_n must be 3D array");
    }
    const ssize_t d0 = nbuf.shape[0];
    const ssize_t d1 = nbuf.shape[1];
    const ssize_t d2 = nbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* n_ptr = static_cast<float*>(nbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double total = 0.0;
            ssize_t m = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float n = n_ptr[base + t];
                if (!std::isnan(n) && n > 0.0f) {
                    total += n;
                    ++m;
                }
            }
            if (total <= 0.0 || m == 0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double H = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float n = n_ptr[base + t];
                if (!std::isnan(n) && n > 0.0f) {
                    double p = n / total;
                    H -= p * std::log(p);
                }
            }
            double norm = std::log(static_cast<double>(m));
            double conc = (norm > 0.0) ? (1.0 - H / norm) : std::numeric_limits<double>::quiet_NaN();
            res_ptr[i * d1 + j] = static_cast<float>(conc);
        }
    }
}

void bind_cube2mat_n_entropy_concentration(py::module& m) {
    m.def("cube2mat_n_entropy_concentration", &cube2mat_n_entropy_concentration,
          py::arg("result"), py::arg("cubes_map"),
          "1 - normalized entropy of trade count distribution across RTH bars.");
}

