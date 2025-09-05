#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_absret_per_volume(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);

    auto cbuf = price_arr.request();
    auto vbuf = vol_arr.request();
    if (cbuf.ndim != 3 || vbuf.ndim != 3) {
        throw std::runtime_error("last_price and interval_volume must be 3D arrays");
    }
    if (cbuf.shape[0] != vbuf.shape[0] || cbuf.shape[1] != vbuf.shape[1] || cbuf.shape[2] != vbuf.shape[2]) {
        throw std::runtime_error("last_price and interval_volume must have same shape");
    }

    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D and match leading dims");
    }

    const float* price_ptr = static_cast<float*>(cbuf.ptr);
    const float* vol_ptr   = static_cast<float*>(vbuf.ptr);
    float* res_ptr         = static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            ssize_t base = i * (d1 * d2) + j * d2;
            float totv = 0.0f;
            float sumabs = 0.0f;
            float prev_close = std::numeric_limits<float>::quiet_NaN();
            bool prev_valid = false;
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                float v = vol_ptr[base + t];
                if (!std::isfinite(c) || !std::isfinite(v)) continue;
                totv += v;
                if (prev_valid) {
                    float r = std::log(c) - std::log(prev_close);
                    sumabs += std::fabs(r);
                }
                prev_close = c;
                prev_valid = true;
            }
            res_ptr[i*d1 + j] = (totv > 0.0f) ? (sumabs / totv) : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_absret_per_volume(py::module& m) {
    m.def("cube2mat_absret_per_volume", &cube2mat_absret_per_volume,
          py::arg("result"), py::arg("cubes_map"),
          "Σ|log returns| divided by total interval_volume over RTH.");
}

