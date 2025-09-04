#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_new_high_count(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                             const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_high")) {
        throw std::runtime_error("cubes_map must contain 'interval_high'");
    }
    auto high_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto hbuf = high_arr.request();
    if (hbuf.ndim != 3) {
        throw std::runtime_error("interval_high must be 3D array");
    }
    const ssize_t d0 = hbuf.shape[0];
    const ssize_t d1 = hbuf.shape[1];
    const ssize_t d2 = hbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* high_ptr = static_cast<float*>(hbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float max_so_far = std::numeric_limits<float>::quiet_NaN();
            int count = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float h = high_ptr[base + t];
                if (std::isnan(h)) continue;
                if (max_so_far==max_so_far && h > max_so_far) ++count;
                if (!(max_so_far==max_so_far) || h > max_so_far) max_so_far = h;
            }
            res_ptr[i * d1 + j] = (max_so_far==max_so_far) ? static_cast<float>(count)
                                                           : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_new_high_count(py::module& m) {
    m.def("cube2mat_new_high_count", &cube2mat_new_high_count,
          py::arg("result"), py::arg("cubes_map"),
          "Number of new intraday highs within 09:30–15:59.");
}

