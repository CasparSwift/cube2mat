#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_accel_align_rate(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> c;
            c.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = price_ptr[base + t];
                if (!std::isnan(v)) c.push_back(static_cast<double>(v));
            }
            if (c.size() < 4) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<double> d1v(c.size() - 1);
            for (size_t t = 1; t < c.size(); ++t) d1v[t-1] = c[t] - c[t-1];
            std::vector<double> d2v(d1v.size() - 1);
            for (size_t t = 1; t < d1v.size(); ++t) d2v[t-1] = d1v[t] - d1v[t-1];
            size_t cnt = 0, tot = 0;
            for (size_t t = 0; t < d2v.size(); ++t) {
                double x = d1v[t+1];
                double y = d2v[t];
                if (x == 0.0 || y == 0.0) continue;
                ++tot;
                if (x * y > 0.0) ++cnt;
            }
            if (tot > 0) {
                res_ptr[i * d1 + j] = static_cast<float>(static_cast<double>(cnt) / tot);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_accel_align_rate(py::module& m) {
    m.def("cube2mat_accel_align_rate", &cube2mat_accel_align_rate,
          py::arg("result"), py::arg("cubes_map"),
          "Fraction of time with curvature pushing in the velocity direction.");
}

