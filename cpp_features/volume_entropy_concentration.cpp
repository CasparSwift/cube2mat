#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_volume_entropy_concentration(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double total = 0.0;
            ssize_t m = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (!std::isnan(v) && v > 0.0f) {
                    total += v;
                    ++m;
                }
            }
            if (total <= 0.0 || m == 0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double H = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (!std::isnan(v) && v > 0.0f) {
                    double p = v / total;
                    H -= p * std::log(p);
                }
            }
            double norm = std::log(static_cast<double>(m));
            double conc = (norm > 0.0) ? (1.0 - H / norm) : std::numeric_limits<double>::quiet_NaN();
            res_ptr[i * d1 + j] = static_cast<float>(conc);
        }
    }
}

void bind_cube2mat_volume_entropy_concentration(py::module& m) {
    m.def("cube2mat_volume_entropy_concentration", &cube2mat_volume_entropy_concentration,
          py::arg("result"), py::arg("cubes_map"),
          "1 - normalized entropy of volume distribution across intraday bars.");
}

