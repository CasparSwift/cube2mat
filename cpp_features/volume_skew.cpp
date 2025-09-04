#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_volume_skew(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                          const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v)) vals.push_back(v);
            }
            const ssize_t n = static_cast<ssize_t>(vals.size());
            if (n < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean = 0.0;
            for (double v : vals) mean += v;
            mean /= static_cast<double>(n);
            double m2 = 0.0, m3 = 0.0;
            for (double v : vals) {
                double d = v - mean;
                m2 += d * d;
                m3 += d * d * d;
            }
            double s2 = m2 / static_cast<double>(n - 1);
            if (!(s2 > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double s = std::sqrt(s2);
            double m3_mean = m3 / static_cast<double>(n);
            double g1 = m3_mean / (s * s * s);
            res_ptr[i * d1 + j] = std::isfinite(g1) ? static_cast<float>(g1)
                                                    : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_volume_skew(py::module& m) {
    m.def("cube2mat_volume_skew", &cube2mat_volume_skew,
          py::arg("result"), py::arg("cubes_map"),
          "Skewness of volume distribution.");
}
