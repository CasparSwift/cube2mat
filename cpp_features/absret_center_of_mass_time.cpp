#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_absret_center_of_mass_time(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    const double span = static_cast<double>(d2 - 1);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double num = 0.0;
            double den = 0.0;
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                double r = std::log(p1 / p0);
                double a = std::fabs(r);
                if (!std::isfinite(a) || a <= 0.0) continue;
                double tf = span > 0 ? static_cast<double>(t) / span : 0.0;
                num += tf * a;
                den += a;
            }
            if (den > 0.0) {
                double val = num / den;
                val = std::min(std::max(val, 0.0), 1.0);
                res_ptr[i * d1 + j] = static_cast<float>(val);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_absret_center_of_mass_time(py::module& m) {
    m.def("cube2mat_absret_center_of_mass_time", &cube2mat_absret_center_of_mass_time,
          py::arg("result"), py::arg("cubes_map"),
          "Weighted time centroid by |log return| within RTH, normalized to [0,1].");
}

