#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_volume_median_to_mean_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v >= 0.0f) vals.push_back(v);
            }
            const ssize_t n = static_cast<ssize_t>(vals.size());
            if (n == 0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean = 0.0;
            for (float v : vals) mean += v;
            mean /= static_cast<double>(n);
            if (!(mean > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::sort(vals.begin(), vals.end());
            double median;
            if (n % 2 == 1) {
                median = vals[n / 2];
            } else {
                median = 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
            }
            double ratio = median / mean;
            res_ptr[i * d1 + j] = std::isfinite(ratio) ? static_cast<float>(ratio)
                                                       : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_volume_median_to_mean_ratio(py::module& m) {
    m.def("cube2mat_volume_median_to_mean_ratio", &cube2mat_volume_median_to_mean_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Median-to-mean ratio of volume.");
}
