#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_katz_fd_close(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                            const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float p0 = price_ptr[base];
            if (std::isnan(p0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double L = 0.0;
            double dmax = 0.0;
            float prev = p0;
            ssize_t n = 1;
            for (ssize_t t = 1; t < d2; ++t) {
                float p = price_ptr[base + t];
                if (std::isnan(p)) continue;
                L += std::fabs(p - prev);
                double dist = std::fabs(p - p0);
                if (dist > dmax) dmax = dist;
                prev = p;
                ++n;
            }
            if (L <= 0.0 || dmax <= 0.0 || n < 2) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double D = std::log(static_cast<double>(n)) /
                        (std::log(static_cast<double>(n)) + std::log(dmax / L));
            res_ptr[i * d1 + j] = static_cast<float>(D);
        }
    }
}

void bind_cube2mat_katz_fd_close(py::module& m) {
    m.def("cube2mat_katz_fd_close", &cube2mat_katz_fd_close,
          py::arg("result"), py::arg("cubes_map"),
          "Katz fractal dimension of the close path during RTH.");
}

