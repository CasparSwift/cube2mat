#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_new_low_count(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                            const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_low")) {
        throw std::runtime_error("cubes_map must contain 'interval_low'");
    }
    auto low_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_low"]);
    auto lbuf = low_arr.request();
    if (lbuf.ndim != 3) {
        throw std::runtime_error("interval_low must be 3D array");
    }
    const ssize_t d0 = lbuf.shape[0];
    const ssize_t d1 = lbuf.shape[1];
    const ssize_t d2 = lbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* low_ptr = static_cast<float*>(lbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float min_so_far = std::numeric_limits<float>::quiet_NaN();
            int count = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float l = low_ptr[base + t];
                if (std::isnan(l)) continue;
                if (min_so_far==min_so_far && l < min_so_far) ++count;
                if (!(min_so_far==min_so_far) || l < min_so_far) min_so_far = l;
            }
            res_ptr[i * d1 + j] = (min_so_far==min_so_far) ? static_cast<float>(count)
                                                           : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_new_low_count(py::module& m) {
    m.def("cube2mat_new_low_count", &cube2mat_new_low_count,
          py::arg("result"), py::arg("cubes_map"),
          "Number of new intraday lows within 09:30–15:59.");
}

