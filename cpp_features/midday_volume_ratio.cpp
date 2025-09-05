#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_midday_volume_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                  const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    const double TOTAL_MIN = 389.0;
    const double interval_min = TOTAL_MIN / static_cast<double>(d2);
    ssize_t start_idx = static_cast<ssize_t>(std::ceil(150.0 / interval_min));
    ssize_t end_idx   = static_cast<ssize_t>(std::floor(205.0 / interval_min));
    if (start_idx < 0) start_idx = 0;
    if (end_idx > d2 - 1) end_idx = d2 - 1;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double sum_all = 0.0, cnt_all = 0.0;
            double sum_mid = 0.0, cnt_mid = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isnan(v)) continue;
                double dv = static_cast<double>(v);
                sum_all += dv; cnt_all += 1.0;
                if (t >= start_idx && t <= end_idx) {
                    sum_mid += dv; cnt_mid += 1.0;
                }
            }
            if (cnt_all > 0 && sum_all > 0.0 && cnt_mid > 0) {
                double mean_all = sum_all / cnt_all;
                double mean_mid = sum_mid / cnt_mid;
                res_ptr[i * d1 + j] = static_cast<float>(mean_mid / mean_all);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_midday_volume_ratio(py::module& m) {
    m.def("cube2mat_midday_volume_ratio", &cube2mat_midday_volume_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Midday (12-13h) mean volume / session mean volume.");
}

