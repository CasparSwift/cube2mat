#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void alpha_example(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                   const py::dict& cubes_map) {
    if (!cubes_map.contains("close") || !cubes_map.contains("volume")) {
        throw std::runtime_error("cubes_map must contain 'close' and 'volume'");
    }

    auto close_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["close"]);
    auto volume_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["volume"]);

    auto cbuf = close_arr.request();
    auto vbuf = volume_arr.request();
    if (cbuf.ndim != 3 || vbuf.ndim != 3) {
        throw std::runtime_error("close and volume must be 3D arrays of shape (d0,d1,d2)");
    }
    if (cbuf.shape[0] != vbuf.shape[0] || cbuf.shape[1] != vbuf.shape[1] || cbuf.shape[2] != vbuf.shape[2]) {
        throw std::runtime_error("close and volume must have the same shape");
    }

    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match close/volume leading dims");
    }

    const float* close_ptr  = static_cast<float*>(cbuf.ptr);
    const float* volume_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr          = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;

            float num = 0.0f;
            float den = 0.0f;

            for (ssize_t t = 0; t < d2; ++t) {
                float v = volume_ptr[base + t];
                float c = close_ptr [base + t];
                if (std::isnan(v) || std::isnan(c) || !(v > 0.0f)) continue;
                num += c * v;
                den += v;
            }

            res_ptr[i * d1 + j] = (den > 0.0f)
                ? (num / den)
                : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_alpha_example(py::module& m) {
    m.def("alpha_example", &alpha_example,
          py::arg("result"), py::arg("cubes_map"),
          "Alpha Example: Compute VWAP over all timesteps for each (i,j).");
}
