#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_u_shape_corr_volume(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<double> v;
            for (ssize_t t = 0; t < d2; ++t) {
                float x = vol_ptr[base + t];
                if (!std::isnan(x)) v.push_back(static_cast<double>(x));
            }
            size_t n = v.size();
            if (n < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean_v = 0.0;
            for (double x : v) mean_v += x;
            mean_v /= n;
            std::vector<double> xv(n);
            double sx = 0.0;
            for (size_t k = 0; k < n; ++k) {
                xv[k] = v[k] - mean_v;
                sx += xv[k] * xv[k];
            }
            if (!(sx > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<double> u(n);
            for (size_t k = 0; k < n; ++k) {
                double t = static_cast<double>(k) / (n - 1);
                u[k] = (t - 0.5);
                u[k] = u[k] * u[k];
            }
            double mean_u = 0.0;
            for (double x : u) mean_u += x;
            mean_u /= n;
            double su = 0.0;
            for (size_t k = 0; k < n; ++k) {
                u[k] -= mean_u;
                su += u[k] * u[k];
            }
            if (!(su > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double dot = 0.0;
            for (size_t k = 0; k < n; ++k) dot += xv[k] * u[k];
            res_ptr[i * d1 + j] = static_cast<float>(dot / (std::sqrt(sx) * std::sqrt(su)));
        }
    }
}

void bind_cube2mat_u_shape_corr_volume(py::module& m) {
    m.def("cube2mat_u_shape_corr_volume", &cube2mat_u_shape_corr_volume,
          py::arg("result"), py::arg("cubes_map"),
          "Correlation of volume with U-shape time template (early/late high).");
}

